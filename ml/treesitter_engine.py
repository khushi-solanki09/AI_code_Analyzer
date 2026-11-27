"""tree-sitter based analysis for JavaScript, Java and C++.

Rules are tree-sitter queries written in the rule configuration. A query
marks the offending node with the capture name `@hit`; the line of that node
is the finding's line. Parsing the original text (not a masked copy) means
the grammar itself decides what is a comment, a string or code.

tree-sitter recovers from syntax errors, but a tree with errors can hide or
invent matches, so the detector falls back to regex rules when `has_error`.
"""

import functools

import tree_sitter_cpp
import tree_sitter_java
import tree_sitter_javascript
from tree_sitter import Language, Parser, Query, QueryCursor

GRAMMARS = {
    "javascript": tree_sitter_javascript.language,
    "java": tree_sitter_java.language,
    "cpp": tree_sitter_cpp.language,
}


@functools.lru_cache(maxsize=None)
def language(lang):
    return Language(GRAMMARS[lang]())


@functools.lru_cache(maxsize=None)
def compile_query(lang, text):
    return Query(language(lang), text)


class Parsed:
    """One parsed source file."""

    def __init__(self, lang, code):
        self.lang = lang
        # a Parser per call: parsers are not safe to share between threads
        self.tree = Parser(language(lang)).parse(code.encode("utf-8"))
        self.has_error = self.tree.root_node.has_error

    def run(self, query_text):
        """Lines of every node captured as @hit."""
        captures = QueryCursor(compile_query(self.lang, query_text)).captures(self.tree.root_node)
        return sorted({node.start_point.row + 1 for node in captures.get("hit", [])})
