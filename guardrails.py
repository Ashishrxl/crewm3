import re

def write_report_guardrail(output):
    try:
        output = output if isinstance(output, str) else output.raw 
    except Exception as e:
        return (False, f"Error retrieving the `raw` argument:\n{str(e)}\n")

    output_lower = output.lower()

    if not re.search(r'#+.*summary', output_lower):
        return (False, "The report must include a Summary section with a header like '## Summary'")

    if not re.search(r'#+.*insights|#+.*recommendations', output_lower):
        return (False, "The report must include an Insights section with a header like '## Insights'")

    if not re.search(r'#+.*citations|#+.*references', output_lower): 
        return (False, "The report must include a Citations (or References) section with a header like '## Citations'") 

    return (True, output)
