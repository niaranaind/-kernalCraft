#include "miniptx/parser.hpp"
#include <stdexcept>

namespace miniptx {

// ============================================================================
// Parser internals
//
// This is a recursive-descent parser with one function per precedence level.
// The idea:
//   - Each precedence level has its own parsing function.
//   - A higher-precedence function is called from within a lower-precedence one.
//   - This naturally handles associativity and precedence without tables.
//
// Our precedence levels (lowest to highest):
//   1. Comparison:       <  >  <=  >=  ==  !=
//   2. Additive:         +  -
//   3. Multiplicative:   *  /
//   4. Unary:            - (prefix)
//   5. Primary:          literals, variables, parenthesized expressions
//
// Error reporting: every error message includes "line N:" so the user can
// find the problem in their source file.
// ============================================================================

class Parser {
public:
    explicit Parser(const std::vector<Token>& tokens)
        : tokens_(tokens), pos_(0) {}

    KernelDecl parse_kernel() {
        int ln = cur().line;
        int col = cur().col;

        // A program is exactly one kernel declaration.
        expect(Tok::Kernel);
        std::string name = expect_ident();
        expect(Tok::LParen);
        auto params = parse_params();
        expect(Tok::RParen);
        int end_ln = cur().line;
        auto body = parse_block(&end_ln);

        // There should be nothing left after the kernel.
        if (cur().kind != Tok::Eof) {
            error("unexpected token after kernel body");
        }

        return {std::move(name), std::move(params), std::move(body), ln, col, end_ln};
    }

private:
    const std::vector<Token>& tokens_;
    size_t pos_;

    // --- Token access helpers ---

    // Return the current token without consuming it.
    const Token& cur() const { return tokens_[pos_]; }

    // Return the current token and advance to the next one.
    Token advance() { return tokens_[pos_++]; }

    // If the current token matches `kind`, consume it and return true.
    bool match(Tok kind) {
        if (cur().kind == kind) { advance(); return true; }
        return false;
    }

    // Consume a token of `kind`, or throw a syntax error.
    Token expect(Tok kind) {
        if (cur().kind != kind) {
            error("expected " + std::string(tok_name(kind)) +
                  ", got " + std::string(tok_name(cur().kind)));
        }
        return advance();
    }

    // Consume an identifier token and return its name.
    std::string expect_ident() {
        Token t = expect(Tok::Ident);
        return t.text;
    }

    // Throw a parse error with the current line number.
    [[noreturn]] void error(const std::string& msg) {
        throw std::runtime_error("line " + std::to_string(cur().line) + ": " + msg);
    }

    // --- Type parsing ---

    // type = "int" ["*"] | "float" ["*"]
    Type parse_type() {
        Type t;
        if (cur().kind == Tok::Int) {
            t.base = BaseType::Int;
            advance();
        } else if (cur().kind == Tok::Float) {
            t.base = BaseType::Float;
            advance();
        } else {
            error("expected type (int or float)");
        }
        // Check for pointer marker.
        if (match(Tok::Star)) {
            t.is_ptr = true;
        }
        return t;
    }

    // --- Parameter list ---

    // param_list = param { "," param }
    // param      = type IDENT
    std::vector<Param> parse_params() {
        std::vector<Param> params;
        if (cur().kind == Tok::RParen) return params;  // empty param list

        do {
            Type t = parse_type();
            std::string name = expect_ident();
            params.push_back({t, std::move(name)});
        } while (match(Tok::Comma));

        return params;
    }

    // --- Block ---

    // block = "{" { stmt } "}"
    std::vector<StmtPtr> parse_block(int* out_end_line = nullptr) {
        expect(Tok::LBrace);
        std::vector<StmtPtr> stmts;
        while (cur().kind != Tok::RBrace && cur().kind != Tok::Eof) {
            stmts.push_back(parse_stmt());
        }
        if (out_end_line) *out_end_line = cur().line;
        expect(Tok::RBrace);
        return stmts;
    }

    // --- Statements ---

    StmtPtr parse_stmt() {
        // A statement starting with a type keyword is a variable declaration.
        if (cur().kind == Tok::Int || cur().kind == Tok::Float) {
            return parse_var_decl();
        }
        if (cur().kind == Tok::If) {
            return parse_if();
        }
        if (cur().kind == Tok::For) {
            return parse_for();
        }
        // Otherwise it must be an assignment: "ident = ..." or "ident[...] = ..."
        if (cur().kind == Tok::Ident) {
            return parse_assign();
        }
        error("expected statement");
    }

    // var_decl = type IDENT "=" expr ";"
    StmtPtr parse_var_decl() {
        int ln = cur().line;
        int col = cur().col;
        Type t = parse_type();
        std::string name = expect_ident();
        expect(Tok::Assign);
        auto init = parse_expr();
        expect(Tok::Semi);
        return make_stmt(VarDeclStmt{t, std::move(name), std::move(init), ln, col});
    }

    // assign_stmt = lvalue "=" expr ";"
    // lvalue = IDENT | IDENT "[" expr "]"
    StmtPtr parse_assign() {
        int ln = cur().line;
        int col = cur().col;
        std::string name = expect_ident();

        // Check for indexed assignment: name[expr] = expr;
        if (match(Tok::LBracket)) {
            auto index = parse_expr();
            expect(Tok::RBracket);
            expect(Tok::Assign);
            auto value = parse_expr();
            expect(Tok::Semi);
            return make_stmt(IndexAssignStmt{
                std::move(name), std::move(index), std::move(value), ln, col});
        }

        // Simple assignment: name = expr;
        expect(Tok::Assign);
        auto value = parse_expr();
        expect(Tok::Semi);
        return make_stmt(AssignStmt{std::move(name), std::move(value), ln, col});
    }

    // Parse an assignment without the trailing semicolon (used in for-loop update).
    // Returns the target name and the value expression.
    std::pair<std::string, ExprPtr> parse_assign_no_semi() {
        std::string name = expect_ident();
        expect(Tok::Assign);
        auto value = parse_expr();
        return {std::move(name), std::move(value)};
    }

    // if_stmt = "if" "(" expr ")" block [ "else" block ]
    StmtPtr parse_if() {
        int ln = cur().line;
        int col = cur().col;
        expect(Tok::If);
        expect(Tok::LParen);
        auto cond = parse_expr();
        expect(Tok::RParen);
        auto then_body = parse_block();

        std::vector<StmtPtr> else_body;
        if (match(Tok::Else)) {
            else_body = parse_block();
        }

        return make_stmt(IfStmt{
            std::move(cond), std::move(then_body), std::move(else_body), ln, col});
    }

    // for_stmt = "for" "(" var_decl expr ";" assign_no_semi ")" block
    //
    // Example: for (int i = 0; i < n; i = i + 1) { ... }
    //
    // The init is a var_decl (which includes its own semicolon).
    // Then the condition expression followed by a semicolon.
    // Then an assignment without semicolon, followed by ")".
    StmtPtr parse_for() {
        int ln = cur().line;
        int col = cur().col;
        expect(Tok::For);
        expect(Tok::LParen);

        // Init: a variable declaration (eats its own ";")
        Type init_type = parse_type();
        std::string init_name = expect_ident();
        expect(Tok::Assign);
        auto init_expr = parse_expr();
        expect(Tok::Semi);

        // Condition
        auto cond = parse_expr();
        expect(Tok::Semi);

        // Update: assignment without semicolon
        auto [upd_name, upd_expr] = parse_assign_no_semi();

        expect(Tok::RParen);
        auto body = parse_block();

        return make_stmt(ForStmt{
            init_type, std::move(init_name), std::move(init_expr),
            std::move(cond),
            std::move(upd_name), std::move(upd_expr),
            std::move(body), ln, col});
    }

    // --- Expressions (recursive descent with one function per precedence level) ---
    //
    // Precedence levels:
    //   Level 1 (lowest):  comparisons    <  >  <=  >=  ==  !=
    //   Level 2:           additive       +  -
    //   Level 3:           multiplicative *  /
    //   Level 4 (highest): unary          - (prefix)
    //   Level 5:           primary        literals, vars, parens
    //
    // Each level parses the next-higher level for its operands, then loops
    // as long as it sees operators at its own level.  This ensures that
    // "a + b * c" parses as "a + (b * c)" because * is parsed at a higher
    // level than +.

    ExprPtr parse_expr() {
        return parse_comparison();
    }

    // cmp_expr = add_expr { ( "<" | ">" | "<=" | ">=" | "==" | "!=" ) add_expr }
    ExprPtr parse_comparison() {
        auto lhs = parse_additive();
        while (true) {
            int ln = cur().line;
            int col = cur().col;
            BinOp op;
            if      (cur().kind == Tok::Lt) op = BinOp::Lt;
            else if (cur().kind == Tok::Gt) op = BinOp::Gt;
            else if (cur().kind == Tok::Le) op = BinOp::Le;
            else if (cur().kind == Tok::Ge) op = BinOp::Ge;
            else if (cur().kind == Tok::Eq) op = BinOp::Eq;
            else if (cur().kind == Tok::Ne) op = BinOp::Ne;
            else break;
            advance();
            auto rhs = parse_additive();
            lhs = make_expr(BinaryExpr{op, std::move(lhs), std::move(rhs), ln, col});
        }
        return lhs;
    }

    // add_expr = mul_expr { ( "+" | "-" ) mul_expr }
    ExprPtr parse_additive() {
        auto lhs = parse_multiplicative();
        while (true) {
            int ln = cur().line;
            int col = cur().col;
            BinOp op;
            if      (cur().kind == Tok::Plus)  op = BinOp::Add;
            else if (cur().kind == Tok::Minus) op = BinOp::Sub;
            else break;
            advance();
            auto rhs = parse_multiplicative();
            lhs = make_expr(BinaryExpr{op, std::move(lhs), std::move(rhs), ln, col});
        }
        return lhs;
    }

    // mul_expr = unary { ( "*" | "/" ) unary }
    ExprPtr parse_multiplicative() {
        auto lhs = parse_unary();
        while (true) {
            int ln = cur().line;
            int col = cur().col;
            BinOp op;
            if      (cur().kind == Tok::Star)  op = BinOp::Mul;
            else if (cur().kind == Tok::Slash) op = BinOp::Div;
            else break;
            advance();
            auto rhs = parse_unary();
            lhs = make_expr(BinaryExpr{op, std::move(lhs), std::move(rhs), ln, col});
        }
        return lhs;
    }

    // unary = "-" unary | primary
    //
    // Unary minus binds tighter than binary *, so "-a * b" parses as
    // "(-a) * b", not "-(a * b)".  The rule is recursive so that
    // "--a" would parse as "-(-a)" (double negation), though that's
    // an unusual thing to write.
    ExprPtr parse_unary() {
        if (cur().kind == Tok::Minus) {
            int ln = cur().line;
            int col = cur().col;
            advance();
            auto operand = parse_unary();
            return make_expr(UnaryMinusExpr{std::move(operand), ln, col});
        }
        return parse_primary();
    }

    // primary = INT_LIT | FLOAT_LIT | IDENT "[" expr "]"
    //         | IDENT "." IDENT | IDENT | "(" expr ")"
    ExprPtr parse_primary() {
        int ln = cur().line;
        int col = cur().col;

        // Integer literal
        if (cur().kind == Tok::IntLit) {
            int val = std::stoi(cur().text);
            advance();
            return make_expr(IntLitExpr{val, ln, col});
        }

        // Float literal
        if (cur().kind == Tok::FloatLit) {
            float val = std::stof(cur().text);
            advance();
            return make_expr(FloatLitExpr{val, ln, col});
        }

        // Identifier — could be a variable, array access, or builtin (tid.x etc.)
        if (cur().kind == Tok::Ident) {
            std::string name = cur().text;
            advance();

            // Array index: name[expr]
            if (cur().kind == Tok::LBracket) {
                advance();
                auto index = parse_expr();
                expect(Tok::RBracket);
                return make_expr(IndexExpr{std::move(name), std::move(index), ln, col});
            }

            // Builtin: tid.x, bid.x, bdim.x
            // Only these three identifiers may appear before a dot.
            if (cur().kind == Tok::Dot) {
                // Validate the base name before consuming the dot.
                if (name != "tid" && name != "bid" && name != "bdim") {
                    throw std::runtime_error(
                        "line " + std::to_string(ln) +
                        ": unknown builtin base '" + name +
                        "'; only tid, bid, bdim are allowed");
                }
                advance();  // consume the dot
                std::string field = expect_ident();

                // Only the .x component is supported.
                if (field != "x") {
                    throw std::runtime_error(
                        "line " + std::to_string(ln) +
                        ": unknown builtin '" + name + "." + field +
                        "'; only .x is supported");
                }
                std::string full = name + "." + field;
                return make_expr(BuiltinExpr{std::move(full), ln, col});
            }

            // Plain variable reference.
            return make_expr(VarExpr{std::move(name), ln, col});
        }

        // Parenthesized expression
        if (match(Tok::LParen)) {
            auto inner = parse_expr();
            expect(Tok::RParen);
            return inner;
        }

        error("expected expression");
    }
};

KernelDecl parse(const std::vector<Token>& tokens) {
    Parser p(tokens);
    return p.parse_kernel();
}

}  // namespace miniptx
