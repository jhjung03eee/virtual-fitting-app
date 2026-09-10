"""Gradio 앱의 배선(wiring)을 GPU 없이 검사한다.

`gradio_app.py` 는 gradio와 tryon_core(torch)를 임포트하므로 로컬에서 실행할 수
없다. 그래서 **소스를 AST로 읽어서** 검사한다.

여기서 잡으려는 것은 딱 하나, **실행해야만 드러나는 배선 실수**다.
`run()` 의 인자 개수와 `click(inputs=[...])` 의 개수가 어긋나면 앱은 멀쩡히
뜨고 버튼을 누르는 순간 터진다. 이 프로젝트에서 인자를 여러 번 추가했고
(scheduler, guidance, seed, 배경 정규화, 신체 치수 5개) 그때마다 두 곳을
같이 고쳐야 했다. 한 번이라도 놓치면 Kaggle에 올려 띄워보기 전까지 모른다.
"""
import ast
import os
import unittest

APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'gradio_app.py')


def load_tree():
    with open(APP_PATH, encoding='utf-8') as f:
        return ast.parse(f.read())


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def find_click_calls(tree):
    """`something.click(...)` 호출을 전부 찾는다."""
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('click', 'change')):
            calls.append(node)
    return calls


def keyword(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


class TestGradioWiring(unittest.TestCase):
    def setUp(self):
        self.tree = load_tree()

    def test_app_parses(self):
        self.assertIsNotNone(self.tree)

    def test_every_handler_input_count_matches_signature(self):
        """click/change의 inputs 개수가 처리 함수의 인자 개수와 같아야 한다.

        어긋나면 앱은 뜨지만 버튼을 누르는 순간 TypeError로 죽는다.
        """
        checked = 0
        for call in find_click_calls(self.tree):
            fn = keyword(call, 'fn')
            inputs = keyword(call, 'inputs')
            if not isinstance(fn, ast.Name) or not isinstance(inputs, ast.List):
                continue
            handler = find_function(self.tree, fn.id)
            if handler is None:
                continue

            expected = len(handler.args.args)
            defaults = len(handler.args.defaults)
            actual = len(inputs.elts)
            self.assertTrue(
                expected - defaults <= actual <= expected,
                f'{fn.id}(): 인자 {expected}개(기본값 {defaults}개)인데 '
                f'inputs 는 {actual}개다. 둘을 같이 고쳐야 한다.',
            )
            checked += 1
        self.assertGreater(checked, 0, 'click 핸들러를 하나도 찾지 못했다')

    def test_outputs_count_matches_return(self):
        """outputs 개수가 처리 함수가 돌려주는 값의 개수와 같아야 한다."""
        for call in find_click_calls(self.tree):
            fn = keyword(call, 'fn')
            outputs = keyword(call, 'outputs')
            if not isinstance(fn, ast.Name) or not isinstance(outputs, ast.List):
                continue
            handler = find_function(self.tree, fn.id)
            if handler is None:
                continue

            returns = [n for n in ast.walk(handler)
                       if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)]
            if not returns:
                continue
            counts = {len(r.value.elts) for r in returns}
            self.assertEqual(
                counts, {len(outputs.elts)},
                f'{fn.id}(): 반환값 {counts}개인데 outputs 는 {len(outputs.elts)}개다',
            )

    def test_handlers_exist(self):
        """fn= 으로 지정한 함수가 실제로 정의돼 있어야 한다."""
        for call in find_click_calls(self.tree):
            fn = keyword(call, 'fn')
            if isinstance(fn, ast.Name):
                self.assertIsNotNone(
                    find_function(self.tree, fn.id),
                    f'{fn.id}() 가 정의돼 있지 않다',
                )

    def test_imports_from_tryon_core_all_exist(self):
        """gradio_app 이 tryon_core 에서 가져오는 이름이 실제로 있는지.

        `from tryon_core import DEFAULT_STEPS` 같은 줄은 앱을 띄우는 순간
        ImportError로 죽는다. 로컬에서는 torch가 없어 gradio_app을 임포트할 수
        없으므로 Kaggle에 올려야 알 수 있었는데, 실제로 그렇게 한 번 터졌다.
        두 파일 모두 AST로 읽으면 GPU 없이 잡을 수 있다.
        """
        core_path = os.path.join(os.path.dirname(APP_PATH), 'tryon_core.py')
        with open(core_path, encoding='utf-8') as f:
            core = ast.parse(f.read())

        defined = set()
        for node in core.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.ImportFrom):
                defined.update(a.asname or a.name for a in node.names)
            elif isinstance(node, ast.Import):
                defined.update((a.asname or a.name).split('.')[0] for a in node.names)

        wanted = [a.name for node in ast.walk(self.tree)
                  if isinstance(node, ast.ImportFrom) and node.module == 'tryon_core'
                  for a in node.names]
        self.assertTrue(wanted, 'gradio_app 이 tryon_core 에서 아무것도 안 가져온다')

        missing = [name for name in wanted if name not in defined]
        self.assertEqual(
            missing, [],
            f'tryon_core 에 없는 이름을 가져온다: {missing}. 앱 시작 시 ImportError가 난다.',
        )

    def test_run_passes_background_option(self):
        """배경 정규화 체크박스가 try_on 까지 실제로 전달되는지.

        UI에 위젯만 추가하고 넘기는 것을 잊으면 체크박스가 아무 일도 안 한다.
        """
        run = find_function(self.tree, 'run')
        self.assertIsNotNone(run)
        names = [a.arg for a in run.args.args]
        self.assertIn('normalize_background', names)

        source = ast.unparse(run)
        self.assertIn('normalize_background=', source,
                      'run() 이 try_on 에 normalize_background 를 넘기지 않는다')


if __name__ == '__main__':
    unittest.main()
