import unittest
from inventory import parse

class ParseTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse('apple=5'), ('apple', 5))

if __name__ == '__main__':
    unittest.main()
