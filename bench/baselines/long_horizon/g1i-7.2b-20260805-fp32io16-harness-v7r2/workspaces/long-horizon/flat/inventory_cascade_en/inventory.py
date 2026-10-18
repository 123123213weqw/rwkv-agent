import unittest
from inventory import parse, total
class ParseTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse('apple=5'), ('apple', 5))
class TotalTest(unittest.TestCase):
    def test_total(self):
        self.assertEqual(total(['apple=5', 'pear=2', 'plum=7']), 14)
if __name__ == '__main__':
    unittest.main()
