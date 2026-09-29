from dataclasses import replace
from fractions import Fraction
from solverpilot import solve, SolveBudget
from solverpilot.runtime.unified import summarize
from solverpilot.runtime.batch import BatchItem
from solverpilot.exact.runtime import ExactSolveResult
from test_total_deadline import problem


def test_validity_proof_and_deadline_are_independent_axes():
    result=solve(problem(),backend='scipy-highs-ds',budget=SolveBudget(wall_time_s=10))
    summary=summarize(result)
    assert summary.feasible and summary.optimality=='independent_numerical_bound'
    assert summary.within_budget and summary.requested_time_s==10
    raw=dict(result.raw_statistics)
    raw['call_budget']={**dict(raw['call_budget']),'within_budget':False,'elapsed_s':11.}
    late=summarize(replace(result,raw_statistics=raw))
    assert late.feasible and late.optimality==summary.optimality and late.within_budget is False
    assert late.status==summary.status  # legacy status is not silently redefined


def test_batch_summary_and_timeout_never_invent_proof():
    item=BatchItem(0,'valid_optimal',[1.],1.,True,'owned',.1,
        independently_verified_optimal=True,problem_data_hash='a'*64,requested_time_s=1.,within_budget=True)
    assert summarize(item).optimality=='independent_numerical_bound'
    stopped=replace(item,status='timeout',x=None,validation_valid=False,independently_verified_optimal=False,
                    within_budget=False)
    assert summarize(stopped).optimality=='not_established' and not summarize(stopped).feasible


def test_exact_bound_and_infeasibility_are_not_exact_optimality():
    bound=ExactSolveResult('bound_verified',True,'finite bound only',(Fraction(2),),Fraction(2),Fraction(1),Fraction(1))
    summary=summarize(bound)
    assert summary.feasible and summary.optimality=='not_established'
    assert summary.termination_evidence=='independent_exact_bound'
    exact=summarize(replace(bound,status='optimal',bound=Fraction(2),absolute_gap=Fraction(0)))
    assert exact.optimality=='independent_exact_certificate'
    infeasible=summarize(ExactSolveResult('infeasible',True,'checked'))
    assert not infeasible.feasible and infeasible.optimality=='not_established'
    assert infeasible.termination_evidence=='independent_exact_infeasibility'
