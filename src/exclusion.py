"""Exclusion rule parsing and filtering for EDL entries.

This module provides functionality to filter out EDL entries based on user-defined
rules in a text file. Rules use a simple expression syntax with field comparisons
and logical operators.

Rule Syntax:
    field_name OPERATOR "value"

Operators:
    IS       - exact match (case-sensitive)
    INCLUDES - substring match (case-sensitive)

Logical operators:
    AND      - both conditions must be true
    OR       - either condition must be true
    NOT      - negates the following expression
    ()       - parentheses for grouping

Lines in the rules file are OR'd together - if ANY line matches, the entry is excluded.

Example rules file:
    file_name IS "" AND name INCLUDES "SYNC"
    name INCLUDES "exclude"
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import EDLEntry


class ExclusionRuleSyntaxError(Exception):
    """Exception raised when an exclusion rule has invalid syntax."""

    def __init__(self, message: str, line_number: int | None = None, line: str | None = None):
        self.line_number = line_number
        self.line = line
        if line_number is not None:
            message = f"Line {line_number}: {message}"
        if line is not None:
            message = f"{message}\n  Rule: {line}"
        super().__init__(message)


class TokenType(Enum):
    """Types of tokens in exclusion rule expressions."""
    FIELD = auto()      # Field name (e.g., "name", "file_name")
    STRING = auto()     # Quoted string value
    IS = auto()         # IS operator
    INCLUDES = auto()   # INCLUDES operator
    AND = auto()        # AND logical operator
    OR = auto()         # OR logical operator
    NOT = auto()        # NOT logical operator
    LPAREN = auto()     # Left parenthesis
    RPAREN = auto()     # Right parenthesis
    EOF = auto()        # End of input


@dataclass
class Token:
    """A token from the exclusion rule lexer."""
    type: TokenType
    value: str
    position: int


# Map field names (English/Dutch, various cases) to EDLEntry attribute names
FIELD_NAME_MAP: dict[str, str] = {
    # name field
    "name": "name",
    "Name": "name",
    "NAME": "name",
    # file_name field
    "file_name": "file_name",
    "FileName": "file_name",
    "filename": "file_name",
    "Filename": "file_name",
    "FILENAME": "file_name",
    "Bestandsnaam": "file_name",
    "bestandsnaam": "file_name",
    "BESTANDSNAAM": "file_name",
    # reel field
    "reel": "reel",
    "Reel": "reel",
    "REEL": "reel",
    # track field
    "track": "track",
    "Track": "track",
    "TRACK": "track",
    # comment field
    "comment": "comment",
    "Comment": "comment",
    "COMMENT": "comment",
}


class Tokenizer:
    """Converts rule text into tokens."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)

    def _skip_whitespace(self) -> None:
        """Skip over whitespace characters."""
        while self.pos < self.length and self.text[self.pos].isspace():
            self.pos += 1

    def _read_string(self) -> str:
        """Read a quoted string value."""
        quote_char = self.text[self.pos]
        self.pos += 1  # Skip opening quote
        start = self.pos

        while self.pos < self.length:
            if self.text[self.pos] == quote_char:
                value = self.text[start:self.pos]
                self.pos += 1  # Skip closing quote
                return value
            self.pos += 1

        raise ExclusionRuleSyntaxError(f"Unterminated string starting at position {start}")

    def _read_identifier(self) -> str:
        """Read an identifier (field name or keyword)."""
        start = self.pos
        while self.pos < self.length and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            self.pos += 1
        return self.text[start:self.pos]

    def tokenize(self) -> list[Token]:
        """Convert the input text into a list of tokens."""
        tokens = []

        while self.pos < self.length:
            self._skip_whitespace()
            if self.pos >= self.length:
                break

            start_pos = self.pos
            char = self.text[self.pos]

            # String literals
            if char in ('"', "'"):
                value = self._read_string()
                tokens.append(Token(TokenType.STRING, value, start_pos))

            # Parentheses
            elif char == '(':
                tokens.append(Token(TokenType.LPAREN, '(', start_pos))
                self.pos += 1
            elif char == ')':
                tokens.append(Token(TokenType.RPAREN, ')', start_pos))
                self.pos += 1

            # Identifiers and keywords
            elif char.isalpha() or char == '_':
                identifier = self._read_identifier()
                upper = identifier.upper()

                if upper == "IS":
                    tokens.append(Token(TokenType.IS, identifier, start_pos))
                elif upper == "INCLUDES":
                    tokens.append(Token(TokenType.INCLUDES, identifier, start_pos))
                elif upper == "AND":
                    tokens.append(Token(TokenType.AND, identifier, start_pos))
                elif upper == "OR":
                    tokens.append(Token(TokenType.OR, identifier, start_pos))
                elif upper == "NOT":
                    tokens.append(Token(TokenType.NOT, identifier, start_pos))
                else:
                    # Must be a field name
                    if identifier not in FIELD_NAME_MAP:
                        raise ExclusionRuleSyntaxError(
                            f"Unknown field name '{identifier}'. "
                            f"Valid fields: {', '.join(sorted(set(FIELD_NAME_MAP.values())))}"
                        )
                    tokens.append(Token(TokenType.FIELD, identifier, start_pos))

            else:
                raise ExclusionRuleSyntaxError(f"Unexpected character '{char}' at position {start_pos}")

        tokens.append(Token(TokenType.EOF, "", self.pos))
        return tokens


# Abstract base class for expression nodes
class Expression(ABC):
    """Base class for expression AST nodes."""

    @abstractmethod
    def evaluate(self, entry: EDLEntry) -> bool:
        """Evaluate the expression against an EDL entry.

        Args:
            entry: The EDL entry to evaluate against

        Returns:
            True if the expression matches the entry
        """
        pass


@dataclass
class Comparison(Expression):
    """A comparison expression: field_name OPERATOR "value"."""
    field: str       # Mapped attribute name on EDLEntry
    operator: str    # "IS" or "INCLUDES"
    value: str       # The string value to compare against

    def evaluate(self, entry: EDLEntry) -> bool:
        field_value = str(getattr(entry, self.field, ""))

        if self.operator == "IS":
            return field_value == self.value
        elif self.operator == "INCLUDES":
            return self.value in field_value
        else:
            return False


@dataclass
class AndExpression(Expression):
    """Logical AND of two expressions."""
    left: Expression
    right: Expression

    def evaluate(self, entry: EDLEntry) -> bool:
        return self.left.evaluate(entry) and self.right.evaluate(entry)


@dataclass
class OrExpression(Expression):
    """Logical OR of two expressions."""
    left: Expression
    right: Expression

    def evaluate(self, entry: EDLEntry) -> bool:
        return self.left.evaluate(entry) or self.right.evaluate(entry)


@dataclass
class NotExpression(Expression):
    """Logical NOT of an expression."""
    operand: Expression

    def evaluate(self, entry: EDLEntry) -> bool:
        return not self.operand.evaluate(entry)


class Parser:
    """Recursive descent parser for exclusion rule expressions.

    Grammar:
        expression := or_expr
        or_expr    := and_expr (OR and_expr)*
        and_expr   := not_expr (AND not_expr)*
        not_expr   := NOT not_expr | primary
        primary    := comparison | '(' expression ')'
        comparison := FIELD (IS | INCLUDES) STRING
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> Token:
        """Get the current token."""
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        """Advance to the next token and return the previous one."""
        token = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def _expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type, raise error if not found."""
        token = self._current()
        if token.type != token_type:
            raise ExclusionRuleSyntaxError(
                f"Expected {token_type.name}, got {token.type.name} at position {token.position}"
            )
        return self._advance()

    def parse(self) -> Expression:
        """Parse the tokens into an expression AST."""
        expr = self._parse_or()

        if self._current().type != TokenType.EOF:
            token = self._current()
            raise ExclusionRuleSyntaxError(
                f"Unexpected token '{token.value}' at position {token.position}"
            )

        return expr

    def _parse_or(self) -> Expression:
        """Parse OR expressions."""
        left = self._parse_and()

        while self._current().type == TokenType.OR:
            self._advance()  # consume OR
            right = self._parse_and()
            left = OrExpression(left, right)

        return left

    def _parse_and(self) -> Expression:
        """Parse AND expressions."""
        left = self._parse_not()

        while self._current().type == TokenType.AND:
            self._advance()  # consume AND
            right = self._parse_not()
            left = AndExpression(left, right)

        return left

    def _parse_not(self) -> Expression:
        """Parse NOT expressions."""
        if self._current().type == TokenType.NOT:
            self._advance()  # consume NOT
            operand = self._parse_not()
            return NotExpression(operand)

        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        """Parse primary expressions (comparisons or parenthesized expressions)."""
        token = self._current()

        # Parenthesized expression
        if token.type == TokenType.LPAREN:
            self._advance()  # consume '('
            expr = self._parse_or()
            self._expect(TokenType.RPAREN)
            return expr

        # Comparison: FIELD (IS | INCLUDES) STRING
        if token.type == TokenType.FIELD:
            field_token = self._advance()
            field_name = FIELD_NAME_MAP[field_token.value]

            op_token = self._current()
            if op_token.type not in (TokenType.IS, TokenType.INCLUDES):
                raise ExclusionRuleSyntaxError(
                    f"Expected IS or INCLUDES after field name, got {op_token.type.name}"
                )
            self._advance()
            operator = "IS" if op_token.type == TokenType.IS else "INCLUDES"

            value_token = self._expect(TokenType.STRING)

            return Comparison(field_name, operator, value_token.value)

        raise ExclusionRuleSyntaxError(
            f"Unexpected token '{token.value}' at position {token.position}, "
            f"expected field name or '('"
        )


def parse_rule(text: str) -> Expression:
    """Parse a single rule line into an expression.

    Args:
        text: The rule text to parse

    Returns:
        An Expression AST node

    Raises:
        ExclusionRuleSyntaxError: If the rule has invalid syntax
    """
    tokenizer = Tokenizer(text)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    return parser.parse()


class ExclusionRuleSet:
    """A collection of exclusion rules.

    Rules are OR'd together - if ANY rule matches, the entry is excluded.
    """

    def __init__(self, rules: list[Expression]):
        self.rules = rules

    def matches(self, entry: EDLEntry) -> bool:
        """Check if any rule matches the entry (meaning it should be excluded).

        Args:
            entry: The EDL entry to check

        Returns:
            True if the entry should be excluded
        """
        return any(rule.evaluate(entry) for rule in self.rules)

    def __len__(self) -> int:
        return len(self.rules)


def load_exclusion_rules(filepath: Path | str) -> ExclusionRuleSet:
    """Load exclusion rules from a file.

    Each non-empty, non-comment line is parsed as a rule.
    Lines starting with # are treated as comments.

    Args:
        filepath: Path to the rules file

    Returns:
        ExclusionRuleSet containing all parsed rules

    Raises:
        ExclusionRuleSyntaxError: If any rule has invalid syntax
        FileNotFoundError: If the file doesn't exist
    """
    filepath = Path(filepath)
    rules = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            try:
                rule = parse_rule(line)
                rules.append(rule)
            except ExclusionRuleSyntaxError as e:
                raise ExclusionRuleSyntaxError(
                    str(e),
                    line_number=line_num,
                    line=line
                ) from None

    return ExclusionRuleSet(rules)


def filter_edl_entries(
    entries: list[EDLEntry],
    rules: ExclusionRuleSet
) -> tuple[list[EDLEntry], list[EDLEntry]]:
    """Filter EDL entries based on exclusion rules.

    Args:
        entries: List of EDL entries to filter
        rules: Exclusion rules to apply

    Returns:
        Tuple of (kept_entries, excluded_entries)
    """
    kept = []
    excluded = []

    for entry in entries:
        if rules.matches(entry):
            excluded.append(entry)
        else:
            kept.append(entry)

    return kept, excluded
