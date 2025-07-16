import ast

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.for_count = 0
        self.while_count = 0
        self.if_count = 0
        self.recursion_count = 0
        self.logarithmic_pattern = 0
        self.max_loop_depth = 0
        self.current_loop_depth = 0
        self.multiple_recursion = 0
        self.loop_and_recursion = 0

    def visit_For(self, node):
        self.for_count += 1
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
        # Detectar recursión dentro de bucle
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                self.loop_and_recursion = 1
        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_While(self, node):
        self.while_count += 1
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
        # Detectar patrón logarítmico
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                if any(isinstance(t, ast.BinOp) and isinstance(t.op, ast.FloorDiv) for t in ast.walk(n)):
                    self.logarithmic_pattern = 1
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                self.loop_and_recursion = 1
        self.generic_visit(node)
        self.current_loop_depth -= 1

    def visit_If(self, node):
        self.if_count += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        rec_calls = 0
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                if n.func.id == node.name:
                    rec_calls += 1
        if rec_calls > 1:
            self.multiple_recursion = 1
        if rec_calls > 0:
            self.recursion_count += 1
        self.generic_visit(node)

def analizar_codigo(codigo):
    tree = ast.parse(codigo)
    analyzer = CodeAnalyzer()
    analyzer.visit(tree)
    return [
        analyzer.for_count,
        analyzer.while_count,
        analyzer.if_count,
        analyzer.recursion_count,
        analyzer.logarithmic_pattern,
        analyzer.max_loop_depth,
        analyzer.multiple_recursion,
        analyzer.loop_and_recursion
    ]