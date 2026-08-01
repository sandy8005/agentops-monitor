# test_input.py
from input_handler import receive_user_input
import json

result = receive_user_input()
print(json.dumps(result, indent=2))