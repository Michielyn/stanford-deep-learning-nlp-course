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
    """quiz1-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_1()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz1-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_1', lambda f: set([choice.lower() for choice in f()]))

class Test_2(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz2-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_2()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz2-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2', lambda f: set([choice.lower() for choice in f()]))

class Test_3a(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz3a-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_3a()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz3a-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_3a', lambda f: set([choice.lower() for choice in f()]))

class Test_3b(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz3b-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_3b()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz3b-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_3b', lambda f: set([choice.lower() for choice in f()]))

class Test_4a(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz4a-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4a()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz4a-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4a', lambda f: set([choice.lower() for choice in f()]))

class Test_4b(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz4b-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4b()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz4b-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4b', lambda f: set([choice.lower() for choice in f()]))

class Test_4c(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz4c-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4c()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz4c-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4c', lambda f: set([choice.lower() for choice in f()]))

class Test_5a(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz5a-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_5a()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz5a-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_5a', lambda f: set([choice.lower() for choice in f()]))

class Test_5b(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz5b-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_5b()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz5b-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_5b', lambda f: set([choice.lower() for choice in f()]))

class Test_5c(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz5c-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_5c()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz5c-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_5c', lambda f: set([choice.lower() for choice in f()]))

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
