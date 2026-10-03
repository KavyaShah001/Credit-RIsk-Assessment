"""HTML evidence report for an annual panel observation."""
from html import escape
import pandas as pd


def render_panel_report(identifier, year, risk, ratios, drivers, card, scenario=None):
    ratio_table = pd.DataFrame({'feature': ratios.index, 'value': ratios.values}).to_html(index=False, escape=True)
    driver_table = drivers.to_html(index=False, escape=True)
    scenario_html = ''
    if scenario:
        scenario_html = '<h2>Income-statement sensitivity</h2><p>' + escape(scenario) + '</p>'
    limitations = ''.join('<li>' + escape(text) + '</li>' for text in card['limitations'])
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Annual credit-risk report</title>
    <style>body{{font:15px/1.5 Arial;max-width:1050px;margin:35px auto;padding:20px;color:#172a3a}}
    table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border:1px solid #ddd;text-align:left}}h1,h2{{color:#184e6f}}</style>
    <h1>Annual credit-risk report</h1><p>Anonymous entity {escape(str(identifier))}; fiscal year {int(year)}.</p>
    <p>Estimated next-fiscal-year bankruptcy proxy: <strong>{float(risk['selected_pd']):.2%}</strong>.
    Project Internal Credit Risk Rating: {escape(str(risk['internal_grade']))}; score {int(risk['risk_score'])}/100.</p>
    <p>Model {escape(card['selected_model'])}; run {escape(card['run_id'])}. Annual grades are project assumptions.
    A reconstructed annual outcome is used; individual event and filing dates are unavailable.</p>
    <h2>Financial ratios and annual features</h2>{ratio_table}
    <h2>Local model sensitivity</h2><p>Signed PD changes when features are replaced by training medians.
    These comparisons describe model behavior and do not establish causes.</p>{driver_table}
    {scenario_html}<h2>Limitations</h2><ul>{limitations}</ul></html>'''
