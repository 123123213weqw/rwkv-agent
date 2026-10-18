import unittest
from inventory import parse
class ParseTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse('apple=5'), ('apple', 5))
        self.assertEqual(parse('banana=3'), ('banana', 3))
        self.assertEqual(parse('cherry=10'), ('cherry', 10))
if __name__ == '__main__':
    unittest.main()
