"""추적 중인 모든 파이썬 파일의 문법을 검사한다.

테스트가 임포트하지 않는 스크립트(`kaggle_*/`, 일부 `scripts/`)는 깨져도
`python -m unittest` 가 통과한다. 실제로 그렇게 깨진 파일을 커밋한 적이 있다.
커밋 전에 이걸 돌린다.

    python scripts/check_syntax.py

깨진 파일이 있으면 종료 코드 1.
"""
import ast
import subprocess
import sys


def tracked_python_files():
    out = subprocess.run(['git', 'ls-files', '*.py'],
                         capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def main():
    broken = []
    files = tracked_python_files()
    for path in files:
        try:
            with open(path, encoding='utf-8') as f:
                ast.parse(f.read(), filename=path)
        except SyntaxError as e:
            broken.append(f'{path}:{e.lineno}: {e.msg}')
        except OSError as e:
            broken.append(f'{path}: 읽을 수 없음 ({e})')

    if broken:
        print(f'문법 오류 {len(broken)}건:')
        for item in broken:
            print(f'  {item}')
        return 1
    print(f'{len(files)}개 파일 문법 이상 없음')
    return 0


if __name__ == '__main__':
    sys.exit(main())
