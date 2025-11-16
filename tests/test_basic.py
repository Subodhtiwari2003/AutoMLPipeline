import os
def test_repo_files_exist():
    assert os.path.exists('run_demo.py')
    assert os.path.exists('requirements.txt')
