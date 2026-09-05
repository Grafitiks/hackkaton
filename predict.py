import os
import sys

# Обеспечение прямого вызова пайплайна инференса через 'python predict.py'
if __name__ == "__main__":
    from predict_submission import main
    main()
