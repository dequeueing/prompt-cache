# Type hoisting
from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import re

import lxml
import lxml.etree
import gc
from typing import List, Union, Dict, cast, Tuple
from tqdm import tqdm
import time
import conversation
import torch
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,

)

from prompt import compact_spaces


def trim_with_padding(text: str, padding: int = 1) -> str:
    pad_str = ' ' * padding
    return pad_str + text.strip() + pad_str


def is_valid_xml_element_name(name: str) -> bool:
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_\-.]*$"
    return bool(re.fullmatch(pattern, name))


# Wrapper around LlamaTokenizer
class Tokenizer:

    def __init__(self, hf_tokenizer: LlamaTokenizer):
        self.hf_tokenizer = hf_tokenizer

    def encode(self, text: str) -> List[int]:
        return self.hf_tokenizer.encode(text, add_special_tokens=False)[0]

    def decode(self, token_ids: List[int]) -> str:
        return self.hf_tokenizer.decode(token_ids, skip_special_tokens=False)

    @property
    def unk_token(self) -> str:
        return self.hf_tokenizer.unk_token

    @property
    def unk_token_id(self) -> int:
        return self.hf_tokenizer.unk_token_id


class Path:
    path: List[str]

    def __init__(self, path: Union[str, List[str]] = None):

        if path is None:
            path = []

        if type(path) == str:
            if '/' in path:
                path = [s.strip() for s in path.split('/')]
            elif len(path) > 0:
                path = [path]
            else:
                path = []

        self.path = path

    def __len(self):
        return len(self.path)

    def __str__(self):
        return '/'.join(self.path)

    @property
    def is_root(self) -> bool:
        return len(self.path) == 0

    @property
    def head(self) -> str:
        return self.path[0]

    @property
    def next(self) -> Path:
        return Path(self.path[1:])


class Element(ABC):
    name: Union[None, str]
    offset: int

    def __init__(self, offset: int, name: str = None):
        self.name = name
        self.offset = offset

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def token_ids(self) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def position_ids(self) -> List[int]:
        raise NotImplementedError


class Parameter(Element):
    length: int
    placeholder_token: int
    _token_ids: List[int]
    _position_ids: List[int]

    def __init__(self,
                 offset: int,
                 spec: lxml.etree.Element,
                 tokenizer: Tokenizer):
        super().__init__(offset)

        self.placeholder_token = tokenizer.unk_token_id
        self._process(spec, tokenizer)

    def _process(self, root: lxml.etree.Element, tokenizer: Tokenizer):

        assert root.tag == "parameter"

        if "name" not in root.attrib:
            raise ValueError(f'Parameter name is missing')

        if "length" not in root.attrib:
            raise ValueError(f'Parameter length (in tokens) is missing')

        if not is_valid_xml_element_name(root.attrib["name"]):
            raise ValueError(f'Parameter name {root.attrib["name"]} is not valid')

        self.name = root.attrib["name"]
        self.length = int(root.attrib['length'])

        self._token_ids = []
        self._position_ids = list(range(self.offset, self.offset + self.length))

        if "scaffold" in root.attrib:

            self._token_ids = tokenizer.encode(root.attrib["scaffold"])

            if len(self._token_ids) > self.length:
                raise ValueError(f'Scaffold for parameter {self.name} is too long')

        self._token_ids += [self.placeholder_token] * (self.length - len(self._token_ids))

    def __len__(self) -> int:
        return self.length

    def token_ids(self) -> List[int]:
        raise self._token_ids

    def position_ids(self) -> List[int]:
        raise self._position_ids


class TokenSequence(Element):
    text: str
    _token_ids: List[int]
    _position_ids: List[int]

    def __init__(self, offset: int, text: str, tokenizer: Tokenizer):
        super().__init__(offset)

        self.text = text
        self._token_ids = tokenizer.encode(text)
        self._position_ids = list(range(self.offset, self.offset + len(self._token_ids)))

    def __len__(self) -> int:
        return len(self._token_ids)

    def token_ids(self) -> List[int]:
        raise self._token_ids

    def position_ids(self) -> List[int]:
        raise self._position_ids


class UnionModule(Element):
    modules: List[Module]
    length: int
    scaffold_name: str

    def __init__(self, offset, spec: lxml.etree.Element, tokenizer: Tokenizer):

        super().__init__(offset)

        self.modules = []
        self.length = 0

        self._process(spec, tokenizer)

    def _process(self, root: lxml.etree.Element, tokenizer: Tokenizer):

        assert root.tag == "union"

        max_len = 0

        for e in root:
            if e.tag != "module":
                raise ValueError("Only <module> tags are allowed in union")

            module = Module(self.offset, e, tokenizer)
            self.modules.append(module)
            max_len = max(max_len, len(module))

        if "scaffold" in root.attrib:
            scaffold = root.attrib["scaffold"]

            if self.select(scaffold) is None:
                raise ValueError(f"Union scaffold {scaffold} is not found in union")
            self.scaffold_name = scaffold

        # if scaffold is empty, set first element as scaffold
        else:
            self.scaffold_name = self.modules[0].name

        self.length = max_len

    def __len__(self) -> int:
        return self.length

    def token_ids(self) -> List[int]:
        raise ValueError("Cannot get token_ids() on union. Try again on its scaffold")

    def position_ids(self) -> List[int]:
        raise ValueError("Cannot get position_ids() on union. Try again on its scaffold")

    def select(self, path: Union[str, Path]) -> Union[Module, None]:
        if type(path) == str:
            path = Path(path)

        if path.is_root:
            raise ValueError("Cannot select root of union")

        for m in self.modules:
            if m.name == path.head:
                if len(path) == 1:
                    return m
                else:
                    return m.select(path.next)
        return None


class Module(Element):
    children: List[Element]
    length: int
    cache: bool
    _is_root: bool
    _contains_union: bool

    def __init__(self,
                 offset: int,
                 spec: Union[str, lxml.etree.Element],
                 tokenizer: Tokenizer,
                 is_root: bool = False):

        super().__init__(offset)

        self.children = []
        self.length = 0
        self.cache = True  # whether to do cache (true by default)
        self._is_root = is_root
        self._contains_union = False

        if type(spec) == str:
            parser = lxml.etree.XMLParser(recover=True)
            spec = lxml.etree.fromstring(spec, parser=parser)

        self._process(spec, tokenizer)

    def _process(self, root: lxml.etree.Element, tokenizer: Tokenizer):

        if self._is_root:
            assert root.tag == "schema"
        else:
            assert root.tag == "module"

        if "name" not in root.attrib:
            raise ValueError("Module name is missing")

        if not is_valid_xml_element_name(root.attrib["name"]):
            raise ValueError(f'Module name {root.attrib["name"]} is not valid')

        if not self._is_root and "cache" in root.attrib:
            self.cache = root.attrib["cache"] == "true"

        self.name = root.attrib["name"]

        offset = self.offset
        self.children = []

        # prefix text
        if root.text is not None:
            text = compact_spaces(root.text)
            if len(text) > 0:
                seq = TokenSequence(offset, text, tokenizer)
                self.children.append(seq)
                offset += len(seq)

        for e in root:
            match e.tag:
                case "module":
                    m = Module(offset, e, tokenizer)
                    self._contains_union = self._contains_union or m._contains_union

                    # check namespace conflicts
                    submodule_names = [c.name for c in self.modules()]
                    if m.name in submodule_names:
                        raise ValueError(f"Module {m.name} is already defined")

                case "union":
                    m = UnionModule(offset, e, tokenizer)
                    self._contains_union = True
                    submodule_names = [c.name for c in self.modules()]
                    for c in m.modules:
                        if c.name in submodule_names:
                            raise ValueError(f"Module {c.name} is already defined")

                case "parameter":
                    if self._is_root:
                        raise ValueError("Parameters are not allowed in schema")

                    m = Parameter(offset, e, tokenizer)

                    parameter_names = [c.name for c in self.parameters()]
                    if m.name in parameter_names:
                        raise ValueError(f"Parameter {m.name} is already defined")

                case _:
                    m = TokenSequence(offset, e.tostring(), tokenizer)

            self.children.append(m)
            offset += len(m)

            # process tailing text
            if e.tail is not None:
                text = compact_spaces(e.tail)
                if len(text) > 0:
                    seq = TokenSequence(offset, text, tokenizer)
                    self.children.append(seq)
                    offset += len(seq)

        self.length = offset

    def __len__(self) -> int:
        return self.length

    def token_ids(self) -> List[int]:
        if self.contains_union():
            raise ValueError("Cannot get token_ids() on module that contains union. Try again on its scaffold")
        else:
            return [e for c in self.children for e in c.token_ids()]

    def position_ids(self) -> List[int]:
        if self.contains_union():
            raise ValueError("Cannot get position_ids() on module that contains union. Try again on its scaffold")
        else:
            return [e for c in self.children for e in c.position_ids()]

    def get_scaffold(self, *paths: Path) -> Scaffold:
        return Scaffold(self, *paths)

    def select(self, path: Union[str, Path]) -> Union[Module, None]:
        if type(path) == str:
            path = Path(path)

        if path.is_root:
            return self

        for p in self.modules():
            if p.name == path.head:
                if len(path) == 1:
                    return p
                else:
                    return p.select(path.next)
        return None

    def modules(self) -> List[Module]:
        modules = []
        for c in self.children:
            if type(c) == Module:
                modules.append(c)
            elif type(c) == UnionModule:
                c = cast(UnionModule, c)
                for m in c.modules:
                    modules.append(m)
        return modules

    def parameters(self) -> List[Parameter]:
        return [cast(Parameter, c) for c in self.children if type(c) == Parameter]

    def token_sequences(self) -> List[TokenSequence]:
        return [cast(TokenSequence, c) for c in self.children if type(c) == TokenSequence]

    def contains_union(self) -> bool:
        return self._contains_union


# Scaffold is a special module that does not contain unions
class Scaffold(Element):
    module: Module
    children: List[Element]

    def __init__(self, module: Module, *paths: Path):
        super().__init__(module.offset, module.name)
        self.module = module

        self._process(*paths)

    def _process(self, *paths: Path):

        # simple case: module is already a scaffold
        if not self.module.contains_union():
            self.children = self.module.children
            return

        self.children = []

        for e in self.module.children:
            if type(e) == UnionModule:
                union = cast(UnionModule, e)
                rel_paths = [n for n in paths if union.select(n.head)]
                unique_names = list(set([n.head for n in rel_paths]))

                if len(unique_names) > 1:
                    raise ValueError(f"Union cannot have multiple names in scaffold")

                if rel_paths is None or len(rel_paths) == 0:
                    selected_module_name = union.scaffold_name
                else:
                    selected_module_name = unique_names[0]

                selected_module = union.select(selected_module_name)
                scaffold = Scaffold(selected_module, *[n.next for n in rel_paths])

                self.children.append(scaffold)

            elif type(e) == Module:
                module = cast(Module, e)
                scaffold = Scaffold(module, *[n.next for n in paths if n.head == module.name])
                self.children.append(scaffold)

            else:
                self.children.append(e)

    def __len__(self):
        return self.module.length

    def token_ids(self) -> List[int]:
        return [e for c in self.children for e in c.token_ids()]

    def position_ids(self) -> List[int]:
        return [e for c in self.children for e in c.position_ids()]

    def select(self, path: Union[str, Path]) -> Union[Scaffold, None]:
        if type(path) == str:
            path = Path(path)

        if path.is_root:
            return self

        for e in self.children:
            if type(e) == Scaffold:
                scaffold = cast(Scaffold, e)
                if scaffold.name == path.head:
                    if len(path) == 1:
                        return scaffold
                    else:
                        return scaffold.select(path.next)
        return None

    # return all token sequences in this scaffold
    def all_token_sequences(self) -> List[TokenSequence]:
        ret = []
        for e in self.children:
            if type(e) == Scaffold:
                ret += cast(Scaffold, e).all_token_sequences()
            elif type(e) == TokenSequence:
                ret.append(cast(TokenSequence, e))
        return ret


# Schema is a root module that cannot contain parameters
class Schema(Module):
    tokenizer: Tokenizer

    def __init__(self, spec: Union[str, lxml.etree.Element], tokenizer: Tokenizer):
        super().__init__(0, spec, tokenizer, is_root=True)

        self.tokenizer = tokenizer
