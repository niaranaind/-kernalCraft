#pragma once
#include <string>
#include <vector>

namespace miniptx {

// Every distinct token the lexer can produce.
// Keywords: Kernel, If, Else, For, Int, Float
// Literals: IntLit, FloatLit
// Identifiers: Ident
// Operators: Plus, Minus, Star, Slash, Lt, Gt, Le, Ge, Eq, Ne, Assign
// Punctuation: LParen, RParen, LBrace, RBrace, LBracket, RBracket, Comma, Semi, Dot
// Sentinel: Eof
enum class Tok {
    // Literals and identifiers
    Ident, IntLit, FloatLit,
    // Keywords
    Kernel, If, Else, For, Int, Float,
    // Punctuation
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Comma, Semi, Dot,
    // Operators
    Assign,   // =
    Plus,     // +
    Minus,    // -
    Star,     // *
    Slash,    // /
    Lt,       // <
    Gt,       // >
    Le,       // <=
    Ge,       // >=
    Eq,       // ==
    Ne,       // !=
    // End of file
    Eof
};

// Returns a human-readable name for a token kind (used in error messages).
const char* tok_name(Tok k);

struct Token {
    Tok kind;
    std::string text;   // The actual characters from the source
    int line;           // 1-based line number where this token starts
};

// Scan the source string into a flat list of tokens.
// The last token is always Eof.
// Throws std::runtime_error on unrecognized characters.
std::vector<Token> lex(const std::string& src);

}  // namespace miniptx
