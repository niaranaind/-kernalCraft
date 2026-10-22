#include "miniptx/lexer.hpp"
#include <stdexcept>

namespace miniptx {

// Human-readable token names for error messages like "expected ';', got 'int'".
const char* tok_name(Tok k) {
    switch (k) {
        case Tok::Ident:    return "identifier";
        case Tok::IntLit:   return "int literal";
        case Tok::FloatLit: return "float literal";
        case Tok::Kernel:   return "'kernel'";
        case Tok::If:       return "'if'";
        case Tok::Else:     return "'else'";
        case Tok::For:      return "'for'";
        case Tok::Int:      return "'int'";
        case Tok::Float:    return "'float'";
        case Tok::LParen:   return "'('";
        case Tok::RParen:   return "')'";
        case Tok::LBrace:   return "'{'";
        case Tok::RBrace:   return "'}'";
        case Tok::LBracket: return "'['";
        case Tok::RBracket: return "']'";
        case Tok::Comma:    return "','";
        case Tok::Semi:     return "';'";
        case Tok::Dot:      return "'.'";
        case Tok::Assign:   return "'='";
        case Tok::Plus:     return "'+'";
        case Tok::Minus:    return "'-'";
        case Tok::Star:     return "'*'";
        case Tok::Slash:    return "'/'";
        case Tok::Lt:       return "'<'";
        case Tok::Gt:       return "'>'";
        case Tok::Le:       return "'<='";
        case Tok::Ge:       return "'>='";
        case Tok::Eq:       return "'=='";
        case Tok::Ne:       return "'!='";
        case Tok::Eof:      return "end of file";
    }
    return "?";
}

std::vector<Token> lex(const std::string& src) {
    std::vector<Token> tokens;
    int line = 1;                 // Current line number (1-based)
    int col = 1;                  // Current column number (1-based)
    size_t i = 0;                 // Current position in the source string
    const size_t n = src.size();

    while (i < n) {
        char c = src[i];

        // --- Skip whitespace, tracking newlines for line numbers ---
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (c == '\n') {
                ++line;
                col = 1;
            } else if (c == '\t') {
                col += 4;
            } else {
                ++col;
            }
            ++i;
            continue;
        }

        // --- Skip // line comments ---
        if (c == '/' && i + 1 < n && src[i + 1] == '/') {
            i += 2;
            col += 2;
            while (i < n && src[i] != '\n') {
                ++i;
                ++col;
            }
            // Don't skip the newline itself; the loop above will count it.
            continue;
        }

        int tok_line = line;
        int tok_col = col;

        // --- Identifiers and keywords ---
        // Start with a letter or underscore, continue with letters, digits, or _.
        if (std::isalpha(c) || c == '_') {
            size_t start = i;
            while (i < n && (std::isalnum(src[i]) || src[i] == '_')) {
                ++i;
                ++col;
            }
            std::string word = src.substr(start, i - start);

            // Check if the identifier is actually a keyword.
            Tok kind = Tok::Ident;
            if      (word == "kernel") kind = Tok::Kernel;
            else if (word == "if")     kind = Tok::If;
            else if (word == "else")   kind = Tok::Else;
            else if (word == "for")    kind = Tok::For;
            else if (word == "int")    kind = Tok::Int;
            else if (word == "float")  kind = Tok::Float;

            tokens.push_back({kind, std::move(word), tok_line, tok_col});
            continue;
        }

        // --- Numeric literals (int or float) ---
        // We scan digits, then check for a '.' to decide int vs float.
        if (std::isdigit(c)) {
            size_t start = i;
            while (i < n && std::isdigit(src[i])) {
                ++i;
                ++col;
            }

            // If there's a dot followed by a digit, it's a float literal.
            if (i < n && src[i] == '.' && i + 1 < n && std::isdigit(src[i + 1])) {
                ++i;  // skip the dot
                ++col;
                while (i < n && std::isdigit(src[i])) {
                    ++i;
                    ++col;
                }
                tokens.push_back({Tok::FloatLit, src.substr(start, i - start), tok_line, tok_col});
            } else {
                tokens.push_back({Tok::IntLit, src.substr(start, i - start), tok_line, tok_col});
            }
            continue;
        }

        // --- Two-character operators ---
        // We check these before single-character operators so that "<=" is not
        // scanned as "<" followed by "=".
        if (i + 1 < n) {
            std::string two = src.substr(i, 2);
            if (two == "<=") { tokens.push_back({Tok::Le, "<=", tok_line, tok_col}); i += 2; col += 2; continue; }
            if (two == ">=") { tokens.push_back({Tok::Ge, ">=", tok_line, tok_col}); i += 2; col += 2; continue; }
            if (two == "==") { tokens.push_back({Tok::Eq, "==", tok_line, tok_col}); i += 2; col += 2; continue; }
            if (two == "!=") { tokens.push_back({Tok::Ne, "!=", tok_line, tok_col}); i += 2; col += 2; continue; }
        }

        // --- Single-character tokens ---
        Tok kind;
        switch (c) {
            case '(': kind = Tok::LParen;   break;
            case ')': kind = Tok::RParen;   break;
            case '{': kind = Tok::LBrace;   break;
            case '}': kind = Tok::RBrace;   break;
            case '[': kind = Tok::LBracket; break;
            case ']': kind = Tok::RBracket; break;
            case ',': kind = Tok::Comma;    break;
            case ';': kind = Tok::Semi;     break;
            case '.': kind = Tok::Dot;      break;
            case '=': kind = Tok::Assign;   break;
            case '+': kind = Tok::Plus;     break;
            case '-': kind = Tok::Minus;    break;
            case '*': kind = Tok::Star;     break;
            case '/': kind = Tok::Slash;    break;
            case '<': kind = Tok::Lt;       break;
            case '>': kind = Tok::Gt;       break;
            default:
                throw std::runtime_error(
                    "line " + std::to_string(line) + ": unexpected character '" +
                    std::string(1, c) + "'");
        }
        tokens.push_back({kind, std::string(1, c), tok_line, tok_col});
        ++i;
        ++col;
    }

    // Always end with an Eof token so the parser can check for end-of-input.
    tokens.push_back({Tok::Eof, "", line, col});
    return tokens;
}

}  // namespace miniptx
