#!/usr/bin/env python
import unittest, random, sys, copy, argparse, inspect
from graderUtil import graded, CourseTestRunner, GradedTestCase
import traceback

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

#########
# TESTS #
#########

class Test_1(GradedTestCase):
  @graded()
  def test_0(self):
    """1-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_1()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """1-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_1', lambda f: set([choice.lower() for choice in f()]))

class Test_2a(GradedTestCase):
  @graded()
  def test_0(self):
    """2a-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2a()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """2a-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2a', lambda f: set([choice.lower() for choice in f()]))

class Test_2b(GradedTestCase):
  @graded()
  def test_0(self):
    """2b-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2b()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """2b-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2b', lambda f: set([choice.lower() for choice in f()]))

class Test_2c(GradedTestCase):
  @graded()
  def test_0(self):
    """2c-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2c()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """2c-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2c', lambda f: set([choice.lower() for choice in f()]))

class Test_2d(GradedTestCase):
  @graded()
  def test_0(self):
    """2d-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2d()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """2d-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2d', lambda f: set([choice.lower() for choice in f()]))

class Test_2e(GradedTestCase):
  @graded()
  def test_0(self):
    """2e-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2e()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """2e-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2e', lambda f: set([choice.lower() for choice in f()]))

class Test_2f(GradedTestCase):
  @graded()
  def test_0(self):
    """2f-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2f()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """2f-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2f', lambda f: set([choice.lower() for choice in f()]))

class Test_2g(GradedTestCase):
  @graded()
  def test_0(self):
    """2g-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2g()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """2g-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2g', lambda f: set([choice.lower() for choice in f()]))

class Test_2h(GradedTestCase):
  @graded()
  def test_0(self):
    """2h-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2h()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """2h-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2h', lambda f: set([choice.lower() for choice in f()]))

class Test_3(GradedTestCase):
  @graded()
  def test_0(self):
    """3-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_3()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """3-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_3', lambda f: set([choice.lower() for choice in f()]))

class Test_4a(GradedTestCase):
  @graded()
  def test_0(self):
    """4a-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4a()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """4a-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4a', lambda f: set([choice.lower() for choice in f()]))

class Test_4b(GradedTestCase):
  @graded()
  def test_0(self):
    """4b-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4b()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """4b-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4b', lambda f: set([choice.lower() for choice in f()]))

class Test_4c(GradedTestCase):
  @graded()
  def test_0(self):
    """4c-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4c()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """4c-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4c', lambda f: set([choice.lower() for choice in f()]))

class Test_4d(GradedTestCase):
  @graded()
  def test_0(self):
    """4d-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4d()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """4d-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4d', lambda f: set([choice.lower() for choice in f()]))

class Test_4e(GradedTestCase):
  @graded()
  def test_0(self):
    """4e-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4e()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """4e-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4e', lambda f: set([choice.lower() for choice in f()]))

class Test_4f(GradedTestCase):
  @graded()
  def test_0(self):
    """4f-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4f()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """4f-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4f', lambda f: set([choice.lower() for choice in f()]))

class Test_4g(GradedTestCase):
  @graded()
  def test_0(self):
    """4g-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4g()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """4g-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4g', lambda f: set([choice.lower() for choice in f()]))

class Test_4h(GradedTestCase):
  @graded()
  def test_0(self):
    """4h-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4h()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """4h-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4h', lambda f: set([choice.lower() for choice in f()]))

class Test_5(GradedTestCase):
  @graded()
  def test_0(self):
    """5-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_5()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """5-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_5', lambda f: set([choice.lower() for choice in f()]))

class Test_6a(GradedTestCase):
  @graded()
  def test_0(self):
    """6a-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6a()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """6a-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6a', lambda f: set([choice.lower() for choice in f()]))

class Test_6b(GradedTestCase):
  @graded()
  def test_0(self):
    """6b-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6b()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """6b-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6b', lambda f: set([choice.lower() for choice in f()]))

class Test_6c(GradedTestCase):
  @graded()
  def test_0(self):
    """6c-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6c()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """6c-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6c', lambda f: set([choice.lower() for choice in f()]))

class Test_6d(GradedTestCase):
  @graded()
  def test_0(self):
    """6d-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6d()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """6d-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6d', lambda f: set([choice.lower() for choice in f()]))

class Test_6e(GradedTestCase):
  @graded()
  def test_0(self):
    """6e-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6e()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """6e-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6e', lambda f: set([choice.lower() for choice in f()]))

class Test_6f(GradedTestCase):
  @graded()
  def test_0(self):
    """6f-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6f()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """6f-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6f', lambda f: set([choice.lower() for choice in f()]))

class Test_6g(GradedTestCase):
  @graded()
  def test_0(self):
    """6g-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6g()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """6g-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6g', lambda f: set([choice.lower() for choice in f()]))

class Test_6h(GradedTestCase):
  @graded()
  def test_0(self):
    """6h-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6h()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """6h-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6h', lambda f: set([choice.lower() for choice in f()]))

class Test_6i(GradedTestCase):
  @graded()
  def test_0(self):
    """6i-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6i()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """6i-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6i', lambda f: set([choice.lower() for choice in f()]))

class Test_6j(GradedTestCase):
  @graded()
  def test_0(self):
    """6j-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6j()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """6j-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6j', lambda f: set([choice.lower() for choice in f()]))

class Test_6k(GradedTestCase):
  @graded()
  def test_0(self):
    """6k-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6k()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """6k-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6k', lambda f: set([choice.lower() for choice in f()]))

class Test_6l(GradedTestCase):
  @graded()
  def test_0(self):
    """6l-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6l()])
    self.assertTrue(response.issubset(set(['t','f'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """6l-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6l', lambda f: set([choice.lower() for choice in f()]))

class Test_7(GradedTestCase):
  @graded()
  def test_0(self):
    """7-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_7()])
    self.assertTrue(response.issubset(set(['a','b'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """7-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_7', lambda f: set([choice.lower() for choice in f()]))

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  # Parse for a specific test
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='grader.py'))
  CourseTestRunner().run(assignment)
