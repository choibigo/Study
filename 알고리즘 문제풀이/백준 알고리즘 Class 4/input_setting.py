import os, sys

def setting():
    root_path = os.path.dirname(os.path.abspath(__file__))
    sys.stdin = open(root_path+"\\input.txt")

print("## Start ##")
setting()