"""Placeholder report writer; replaced below."""
import os
def write_report(results, out):
    p = os.path.join(out, "report.md")
    open(p, "w").write("placeholder\n")
    return p
