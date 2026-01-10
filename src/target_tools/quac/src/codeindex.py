#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import itertools
from typing import cast, Sequence, Callable

import libcst as cst
import libcst.matchers as cstm
from libcst.metadata import MetadataWrapper, PositionProvider
from libcst import ParserSyntaxError


@dataclass(eq=True, order=True)
class Position:
    """Source position for a code element."""
    line_number: int
    col_offset: int

    @staticmethod
    def from_item(it: dict) -> "Position":
        return Position(it.get("line_number", -1), it.get("col_offset", -1))

    def to_item(self) -> dict[str, int]:
        return { "line_number": self.line_number, "col_offset": self.col_offset }

    def __repr__(self) -> str:
        return f"({self.line_number}, {self.col_offset})"


class FuncIndex:
    __slots__ = ("qualname", "pos", "params", "vars")
    def __init__(self, qualname: str, position: Position):
        self.qualname = qualname
        self.pos = position
        self.params: dict[str, Position] = {}
        self.vars: dict[str, list[Position]] = defaultdict(list)

    def print(self) -> None:
        print(f"function {self.qualname} {self.pos}")
        for param, pos in self.params.items():
            print(f"    param {param} {pos}")
        for var, pos_list in self.vars.items():
            print(f"    var {var} {pos_list}")

    def from_match(self, it: dict, match: Callable[[str, Position], bool]) -> dict:
        """Returns 'it' with an updated description if 'match' is true for an element."""

        # match the function (return value)?
        if match(self.qualname, self.pos):
            return change_it(it, {
                "function": self.qualname,
            } | self.pos.to_item())

        # match a parameter?
        if (par_pos := next(
            (
                (p, p_pos)
                for p, p_pos in self.params.items()
                if match(p, p_pos)
            ),
            None)
        ):
            return change_it(it, {
                "function": self.qualname,
                "parameter": par_pos[0],
            } | par_pos[1].to_item())

        # match a variable?
        if (var_pos := next(
            (
                (v, p)
                for v, pos_list in self.vars.items()
                for p in pos_list
                if match(v, p)
            ),
            None)
        ):
            return change_it(it, {
                "function": self.qualname,
                "variable": var_pos[0] + it_subscript(it),
            } | var_pos[1].to_item())

        # didn't match.
        return {}


class ModuleIndex(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    @staticmethod
    def from_source(source: str) -> "ModuleIndex":
        wrapper = MetadataWrapper(cst.parse_module(source))
        idx = ModuleIndex()
        wrapper.visit(idx)
        return idx

    def __init__(self) -> None:
        self.functions: dict[str, FuncIndex] = {}
        self.lambdas: list[FuncIndex] = []
        self.comprehensions: list[FuncIndex] = []
        self.module_vars: dict[str, list[Position]] = defaultdict(list)

        self._scope_stack: list[str] = []
        self._func_stack: list[FuncIndex] = []
        self._comp_stack: list[FuncIndex] = []

    def print(self) -> None:
        for func in itertools.chain(self.functions.values(), self.lambdas, self.comprehensions):
            func.print()
        for var, pos_list in self.module_vars.items():
            print(f"var {var} {pos_list}")

    def _qualify(self, name: str) -> str:
        return ".".join([*self._scope_stack, name]) if self._scope_stack else name

    def _position(self, node: cst.CSTNode) -> Position:
        cr = self.get_metadata(PositionProvider, node)
        return Position(cr.start.line, cr.start.column+1) # columns start at 1

    def visit_ClassDef(self, node: cst.ClassDef):
        self._scope_stack.append(node.name.value)
        return True

    def leave_ClassDef(self, node: cst.ClassDef):
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef):
        qual = self._qualify(node.name.value)
        finfo = FuncIndex(qual, self._position(node.name))
        self.functions[qual] = finfo
        self._scope_stack.append(node.name.value)
        self._func_stack.append(finfo)

        for p in cast(Sequence[cst.Param], cstm.findall(node.params, cstm.Param())):
            finfo.params[p.name.value] = self._position(p.name)
        return True

    def leave_FunctionDef(self, node: cst.FunctionDef):
        self._func_stack.pop()
        self._scope_stack.pop()

    def visit_Lambda(self, node: cst.Lambda):
        finfo = FuncIndex('lambda', self._position(node))
        self.lambdas.append(finfo)
        self._func_stack.append(finfo) # any assignment targets (variables) stay within the lambda
        for p in cast(Sequence[cst.Param], cstm.findall(node.params, cstm.Param())):
            finfo.params[p.name.value] = self._position(p.name)
        return True

    def leave_Lambda(self, node: cst.Lambda):
        self._func_stack.pop()

    def _start_comprehension(self, pos: Position) -> None:
        finfo = FuncIndex('<comprehension>', pos)
        self.comprehensions.append(finfo)
        self._comp_stack.append(finfo)
        # note _func_stack stays the same -- any assignment targets (variables) go to the current function

    def visit_ListComp(self, node: cst.ListComp):
        self._start_comprehension(self._position(node))
        return True

    def leave_ListComp(self, node: cst.ListComp):
        self._comp_stack.pop()

    def visit_SetComp(self, node: cst.SetComp):
        self._start_comprehension(self._position(node))
        return True

    def leave_SetComp(self, node: cst.SetComp):
        self._comp_stack.pop()

    def visit_DictComp(self, node: cst.DictComp):
        self._start_comprehension(self._position(node))
        return True

    def leave_DictComp(self, node: cst.DictComp):
        self._comp_stack.pop()

    def visit_GeneratorExp(self, node: cst.GeneratorExp):
        self._start_comprehension(self._position(node))
        return True

    def leave_GeneratorExp(self, node: cst.GeneratorExp):
        self._comp_stack.pop()

    def visit_CompFor(self, node: cst.CompFor):
        # In Python >= 3.8, comprehensions create their own scope;
        # temporarily have any targets (variables) created by 'for' go there
        self._func_stack.append(self._comp_stack[-1])
        self._collect_targets(node.target)
        self._func_stack.pop()

    def _record_var_token(self, node: cst.CSTNode, token_text: str):
        if self._func_stack:
            self._func_stack[-1].vars[token_text].append(self._position(node))
        else:
            self.module_vars[self._qualify(token_text)].append(self._position(node))

    def _collect_targets(self, target: cst.CSTNode):
        if isinstance(target, (cst.Name, cst.Attribute, cst.Subscript)):
            self._record_var_token(target, cast(str, cst.helpers.get_full_name_for_node(target)))
        elif isinstance(target, (cst.Tuple, cst.List)):
            for e in target.elements:
                if e is not None and e.value is not None:
                    self._collect_targets(e.value)
        elif isinstance(target, cst.StarredElement):
            if target.value is not None:
                self._collect_targets(target.value)

    def visit_Assign(self, node: cst.Assign):
        for t in node.targets:
            self._collect_targets(t.target)
        return True

    def visit_NamedExpr(self, node: cst.NamedExpr):
        self._collect_targets(node.target)
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign):
        self._collect_targets(node.target)
        return True

    def visit_AugAssign(self, node: cst.AugAssign):
        self._collect_targets(node.target)
        return True

    def visit_For(self, node: cst.For):
        self._collect_targets(node.target)
        return True

    def visit_With(self, node: cst.With):
        for item in node.items:
            if item.asname and item.asname.name:
                self._collect_targets(item.asname.name)
        return True

    def visit_ExceptHandler(self, node: cst.ExceptHandler):
        if node.name:
            self._collect_targets(node.name)
        return True

#    def visit_Call(self, node: cst.Call):
#        self._collect_targets(node.func)
#        return True

    def from_desc(self, it: dict) -> list[Position]:
        """Returns the list of positions from an item's description."""
        func = self.functions.get(it.get("function"))
        
        if "parameter" in it:
            if not func: return []
            if not (pos := func.params.get(it.get("parameter"))): return []
            return [pos]

        if "variable" in it:
            if "function" in it:
                if not func: return []
                d = func.vars
            else:
                d = self.module_vars

            it_variable = it.get("variable")
            varname = it_variable.split('[')[0] if '[' in it_variable else it_variable
            return d.get(varname, [])

        if not func: return []
        return [func.pos]

    def is_lambda_pos(self, pos: Position) -> bool:
        return any(
            pos == l.pos
            or pos in l.params.values()
            or any(pos in pos_list for pos_list in l.vars.values())
            for l in self.lambdas
        )

    def from_match(self, it: dict, match: Callable[[str, Position], bool]) -> dict:
        """Returns 'it' with an updated description if 'match' is true for an element."""

        for func in itertools.chain(self.functions.values(), self.lambdas):
            if (desc := func.from_match(it, match)):
                return desc

        # match a module variable?
        if (var_pos := next(
            (
                (v, p)
                for v, pos_list in self.module_vars.items()
                for p in pos_list
                if match(v, p)
            ),
            None)
        ):
            return change_it(it, {
                "variable": var_pos[0] + it_subscript(it),
            } | var_pos[1].to_item())

        # didn't match.
        return {}

    def from_line(self, line: int, it: dict) -> dict:
        return self.from_match(it, lambda name, p: p.line_number == line)

    def from_pos(self, pos: Position, it: dict) -> dict:
        return self.from_match(it, lambda name, p: p == pos)

    def comp_match(self, match: Callable[[str, Position], bool]) -> bool:
        return any(
            match(v, pos)
            for comp in self.comprehensions
            for v, pos_list in comp.vars.items()
            for pos in pos_list
        )

    def in_comp(self, pos: Position) -> bool:
        return self.comp_match(lambda n, p: p == pos)


def it_subscript(it) -> str:
    it_variable = it.get("variable", "")
    return it_variable[it_variable.index('['):] if '[' in it_variable else ''


def change_it(it: dict, desc: dict) -> dict:
    """Updates 'it' with the changes in 'd', trying hard to keep the order
       the same to minimize the differences in the ground truth files."""
    return {
        k: v
        for k, v in (it | desc).items()
        if (k not in ('function', 'parameter', 'variable')) or (k in desc)
        if k != 'type'
    } | {
        k: desc[k] if k in desc else it[k]
        for k in ('type',)
        if k in it or k in desc
    }



def main():
    import argparse
    import re
    ap = argparse.ArgumentParser(description="Index code elements.")
    ap.add_argument("file", help="Python source file to index")
    args = ap.parse_args()

    src = Path(args.file).resolve().read_text(encoding="utf-8")
    src = re.sub(r"<value\d+>", "0", src)
    idx = ModuleIndex.from_source(src)
    idx.print()


if __name__ == "__main__":
    import sys
    sys.exit(main())
