import re

regex = r"^\s*(={2,10}).*\1\s*$"

test_str = " ==Heading==\n==Another Heading==\n===Subheading===\n====Subsubheading====\n==Heading==\n"

regex = re.compile(regex, re.MULTILINE)

# for matchNum, match in enumerate(matches, start=1):
#     print("Match {matchNum} was found at {start}-{end}: {match}".format(
#         matchNum=matchNum, start=match.start(), end=match.end(), match=match.group()
#     ))


def extract_tree(page):
    stack = []
    matches = regex.finditer(page)

    for match in matches:
        level = match.group().count("=") // 2
        title = match.group().strip()[level:-level].strip()
        while len(stack) != 0 and stack[-1][1] >= level:
            stack.pop()

        stack.append((title, level))
        print(level, stack)

    return stack

from utils import scroll_pages

input_file = "subsample_cleaner.xml"
with open(input_file, "r") as f:
    for page in scroll_pages(f):
        extract_tree(page)
        break
        input()