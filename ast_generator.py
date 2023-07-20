from sly import Lexer, Parser


class CalcLexer(Lexer):
    # Set of token names.   This is always required
    tokens = {NUMBER, ID, WHILE, IF, ELSE, PRINT,
              PLUS, MINUS, TIMES, DIVIDE, ASSIGN,
              EQ, LT, LE, GT, GE, NE, FUNC, LET, LPAREN,
              RPAREN, LBRACE, RBRACE, SEMICOLON, COMMA, RETURN, NOT,
              TRUE, FALSE}

    # String containing ignored characters
    ignore = ' \t'

    # Regular expression rules for tokens
    PLUS = r'\+'
    MINUS = r'-'
    TIMES = r'\*'
    DIVIDE = r'/'
    EQ = r'=='
    ASSIGN = r'='
    LE = r'<='
    LT = r'<'
    GE = r'>='
    GT = r'>'
    NE = r'!='
    NOT = r'!'
    SEMICOLON = r';'
    COMMA = r','
    LPAREN = r'\('
    RPAREN = r'\)'
    LBRACE = r'\{'
    RBRACE = r'\}'

    @_(r'\d+')
    def NUMBER(self, t):
        t.value = int(t.value)
        return t

    # Identifiers and keywords
    ID = r'[a-zA-Z_][a-zA-Z0-9_]*'
    ID['if'] = IF
    ID['else'] = ELSE
    ID['while'] = WHILE
    ID['let'] = LET
    ID['fn'] = FUNC
    ID['return'] = RETURN
    ID['true'] = TRUE
    ID['false'] = FALSE

    ignore_comment = r'\#.*'

    # Line number tracking
    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        print('Line %d: Bad character %r' % (self.lineno, t.value[0]))
        self.index += 1

    def __init__(self):
        self.nesting_level = 0


class Node:
    pass


class BinOpExpression(Node):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right


class Number(Node):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"{type(self).__name__}(value={self.value})"


class LetStatement(Node):
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name}, expr={self.expr})"


class ReturnStatement(Node):
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"{type(self).__name__}(expr={self.expr})"


class IfStatement(Node):
    def __init__(self, condition, consequence, alternative):
        self.condition = condition
        self.consequence = consequence
        self.alternative = alternative

    def __repr__(self):
        return f"{type(self).__name__}(condition={self.condition}," \
               f" consequence={self.consequence}, alternative={self.alternative})"


class FuncLiteral(Node):
    def __init__(self, params, statements):
        self.params = params
        self.statements = statements

    def __repr__(self):
        return f"{type(self).__name__}(params={self.params}," \
               f" statements={self.statements})"


class CallExpression(Node):
    def __init__(self, identifier, params):
        self.identifier = identifier
        self.params = params

    def __repr__(self):
        return f"{type(self).__name__}(identifier={self.identifier}, params={self.params})"


class Boolean(Node):
    def __init__(self, value):
        self.value = value


class CalcParser(Parser):
    debugfile = 'parser1.out'
    tokens = CalcLexer.tokens

    precedence = (
        ('left', EQ, NE),
        ('left', LT, GT),
        ('left', PLUS, MINUS),
        ('left', TIMES, DIVIDE),
        ('right', UMINUS, NOT),  # Unary minus operator
    )

    @_('statements statement')
    def statements(self, p):
        return p.statements + [p.statement]

    @_('statement')
    def statements(self, p):
        return [p.statement]

    @_('blk_statements')
    def statements(self, p):
        return p.blk_statements

    @_('LBRACE statements RBRACE')
    def blk_statements(self, p):
        return p.statements

    @_('let_statement', 'return_statement', 'if_statement')
    def statement(self, p):
        return p[0]

    @_('ID LPAREN exprlist RPAREN')
    def statement(self, p):
        return CallExpression(p.ID, p.exprlist)

    @_('LET ID ASSIGN expr SEMICOLON')
    def let_statement(self, p):
        # return ('let', p.ID, p.expr)
        return LetStatement(p.ID, p.expr)

    @_('RETURN expr SEMICOLON')
    def return_statement(self, p):
        # return ('return', p.expr)
        return ReturnStatement(p.expr)

    @_('IF expr statements ELSE statements')
    def if_statement(self, p):
        # return ('if', p.expr, p.statements0, p.statements1)
        return IfStatement(condition=p.expr, consequence=p.statements0, alternative=p.statements1)

    @_('FUNC LPAREN params RPAREN statements')
    def statement(self, p):
        # return ('function', p.params, p.statements)
        return FuncLiteral(p.params, p.statements)

    @_('params COMMA param')
    def params(self, p):
        return p.params + [p.param]

    @_('param')
    def params(self, p):
        return [p.param]

    @_('ID')
    def param(self, p):
        return p.ID

    @_('ID LPAREN exprlist RPAREN')
    def expr(self, p):
        return CallExpression(p.ID, p.exprlist)

    @_('exprlist COMMA expr')
    def exprlist(self, p):
        return p.exprlist + [p.expr]

    @_('expr')
    def exprlist(self, p):
        return [p.expr]

    @_('expr PLUS expr', 'expr MINUS expr')
    def expr(self, p):
        # return (p[1], p.expr0, p.expr1)
        return BinOpExpression(p[1], p.expr0, p.expr1)

    @_('expr TIMES expr', 'expr DIVIDE expr')
    def expr(self, p):
        # return (p[1], p.expr0, p.expr1)
        return BinOpExpression(p[1], p.expr0, p.expr1)

    @_('expr LT expr', 'expr GT expr',
       'expr EQ expr', 'expr NE expr')
    def expr(self, p):
        # return (p[1], p.expr0, p.expr1)
        return BinOpExpression(p[1], p.expr0, p.expr1)

    @_('LPAREN expr RPAREN')
    def expr(self, p):
        # return ('grouped-expression', p.expr)
        return p.expr

    @_('MINUS expr %prec UMINUS')
    def expr(self, p):
        # return ('-', p.expr)
        return -p.expr

    @_('NOT expr')
    def expr(self, p):
        # return ('!', p.expr)
        return not p.expr

    @_('TRUE', 'FALSE')
    def expr(self, p):
        # return ('Bool', p[0])
        return Boolean(p[0])

    @_('NUMBER')
    def expr(self, p):
        # return ('num', p.NUMBER)
        return Number(p.NUMBER)

    @_('ID')
    def expr(self, p):
        # return ('id', p.ID)
        return p.ID


if __name__ == '__main__':
    data = '''
       let five = add(5, 10);
'''
    lexer = CalcLexer()
    for tok in lexer.tokenize(data):
        print(tok)

    parser = CalcParser()
    print(parser.parse(lexer.tokenize(data)))
