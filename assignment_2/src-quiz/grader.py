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
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
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
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz2-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_2', lambda f: set([choice.lower() for choice in f()]))

class Test_3(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz3-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_3()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz3-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_3', lambda f: set([choice.lower() for choice in f()]))

class Test_4(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz4-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_4()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz4-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_4', lambda f: set([choice.lower() for choice in f()]))

class Test_5(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz5-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_5()])
    self.assertTrue(response.issubset(set(['a','b','c','d'])), msg='Checks that the response contains only the options available.')
    self.assertEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz5-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_5', lambda f: set([choice.lower() for choice in f()]))

class Test_6(GradedTestCase):
  @graded()
  def test_0(self):
    """quiz6-0-helper:  Sanity check."""
    response = set([choice.lower() for choice in submission.multiple_choice_6()])
    self.assertTrue(response.issubset(set(['a','b','c'])), msg='Checks that the response contains only the options available.')
    self.assertGreaterEqual(len(response),1, msg='Checks that the response is within the cardinality of possible options.')

  @graded(is_hidden=True, after_published=False, hide_errors=True)
  def test_1(self):
    """quiz6-1-hidden:  Multiple choice response."""
    self.compare_with_solution_or_wait(submission, 'multiple_choice_6', lambda f: set([choice.lower() for choice in f()]))

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
