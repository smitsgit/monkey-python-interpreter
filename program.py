# -- sample
code = '''
let five = 5; let ten = 10;
let add = fn(x, y) { x + y;
};
let result = add(five, ten);
'''

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


# -- Grammer
'''
program : statements 

statements : statements statement 
           | statement 

statement : let_statement  | if_statement | return_statement

let_statement: LET ID EQ expr

if_statement: if (expr) LBRACE statements RBRACE else LBRACE statements RBRACE         

'''


class CalcParser(Parser):
    debugfile = 'parser.out'
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

    # @_('let_statement')
    # def statement(self, p):
    #     return [p.statement]
    #
    # @_('return_statement')
    # def statement(self, p):
    #     return [p.statement]
    #
    # @_('if_statement')
    # def statement(self, p):
    #     return [p.statement]

    @_('LET ID ASSIGN expr SEMICOLON')
    def statement(self, p):
        return ('let', p.ID ,p.expr)

    @_('RETURN expr SEMICOLON')
    def statement(self, p):
        return ('return', p.expr)

    @_('IF expr statements ELSE statements')
    def statement(self, p):
        return ('if', p.expr, p.statements0, p.statements1)

    @_('expr PLUS expr', 'expr MINUS expr')
    def expr(self, p):
        return (p[1], p.expr0, p.expr1)

    @_('expr TIMES expr', 'expr DIVIDE expr')
    def expr(self, p):
        return (p[1], p.expr0, p.expr1)

    @_('expr LT expr', 'expr GT expr',
       'expr EQ expr', 'expr NE expr')
    def expr(self, p):
        return (p[1], p.expr0, p.expr1)

    @_('LPAREN expr RPAREN')
    def expr(self, p):
        return ('grouped-expression', p.expr)

    @_('MINUS expr %prec UMINUS')
    def expr(self, p):
        return ('-', p.expr)

    @_('NOT expr')
    def expr(self, p):
        return ('!', p.expr)

    @_('TRUE', 'FALSE')
    def expr(self, p):
        return ('Bool', p[0])

    @_('NUMBER')
    def expr(self, p):
        return ('num', p.NUMBER)

    @_('ID')
    def expr(self, p):
        return ('id', p.ID)


if __name__ == '__main__':
    data = '''
       if (3 < 5) { let five = 5; return 5; } else { return 3;}
'''
    lexer = CalcLexer()
    for tok in lexer.tokenize(data):
        print(tok)

    parser = CalcParser()
    print(parser.parse(lexer.tokenize(data)))

# ('let', ('num', 5), 'return', ('num', 5))
# [('let', ('num', 5)), ('return', ('num', 5))]

# group let five = (5 + 4) * 2;
# [('let', ('*', ('+', ('num', 5), ('num', 4)), ('num', 2))), ('return', ('num', 5))]

# regular let five = 5 + 4 * 2;
# [('let', ('+', ('num', 5), ('*', ('num', 4), ('num', 2)))), ('return', ('num', 5))]

