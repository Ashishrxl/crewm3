import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("agg")

from crewai.tools import BaseTool
from crewai import LLM
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class ChartGeneratorTool(BaseTool):
    name: str = "Create custom plots"
    description: str = (
        "This a tool for automatically creating custom plots based on a research result. "
        "This tool automatically generates plots from a text input, which should have fact checked information. "
        "Pass the full validated information gathered so far as a string."
    )

    def _run(self, research: str) -> str:
        try:
            extraction_prompt = f"""
            You are an expert data visualization assistant. Analyze the provided research text and identify meaningful, insightful charts that can be created to visualize quantifiable data supporting the research's key insights and findings. Only suggest charts for data that includes numerical values, measurable trends, comparisons, or categorical distributions that can be effectively plotted.

            Focus on creating visualizations that highlight trends, comparisons, distributions, or relationships that add value to the research. Avoid suggesting charts for purely qualitative or non-quantifiable information.

            For each chart, provide a JSON object with:
              - "chart_type" (string: choose from "line" for trends over time/continuous, "bar" for comparisons, "histogram" for distributions, "scatter" for relationships, "pie" for proportions)
              - "x_axis" (string: variable name for x-axis, e.g., "year", "category")
              - "y_axis" (string: variable name for y-axis, e.g., "value", "count")
              - "color" (string: optional variable for color grouping/hue, or null if not applicable)
              - "Title" (string: descriptive, insightful title that explains what the chart shows)
              - "data" (dictionary: keys matching x_axis, y_axis, and color variables; values as lists of extracted numerical/categorical data from the research)

            Ensure data is accurately extracted and formatted as lists. If a variable has multiple series (e.g., for color), include all in the data dictionary.

            If no quantifiable data suitable for meaningful visualization is present in the research, return an empty array [].

            Text:
            {research}

            Example output (return valid JSON only):
            [
              {{"chart_type": "line", "x_axis": "year", "y_axis": "funding_amount", "color": "sector", "Title": "AI Research Funding Trends by Sector", "data": {{"year": [2020, 2021, 2022], "funding_amount": [2.5, 3.8, 5.2], "sector": ["Healthcare", "Finance", "Tech"]}}}},
              {{"chart_type": "bar", "x_axis": "tool_name", "y_axis": "adoption_rate", "color": null, "Title": "Market Adoption Rates of AI Tools", "data": {{"tool_name": ["ToolA", "ToolB", "ToolC"], "adoption_rate": [45, 67, 23]}}}}
            ]

            Return only the JSON array, no additional text or explanations.
            """

            # Initialize Gemini model via CrewAI LLM wrapper
            llm = LLM(model="gemini/gemini-2.0-flash")
            llm_response = llm.call([{"role": "user", "content": extraction_prompt}])

            # Clean the response to extract JSON
            llm_response = str(llm_response).strip()
            if llm_response.startswith('```json'):
                llm_response = llm_response[7:]
            if llm_response.endswith('```'):
                llm_response = llm_response[:-3]
            llm_response = llm_response.strip()

            charts_data = json.loads(llm_response)

            if not isinstance(charts_data, list) or len(charts_data) == 0:
                return "No information found in the research to visualize."

            plots_created = []

            for i, chart_info in enumerate(charts_data):
                try:
                    chart_type = str(chart_info.get("chart_type", "")).lower()
                    x_axis = chart_info.get("x_axis", "x")
                    y_axis = chart_info.get("y_axis", "y") 
                    title = chart_info.get("Title", f"Chart {i+1}")
                    hue = chart_info.get("color", None)
                    data = chart_info.get("data", {})

                    df = pd.DataFrame(data)

                    if df.empty:
                        continue

                    plt.figure(figsize=(10, 6))

                    if chart_type == "line":
                        sns.lineplot(data=df, x=x_axis, y=y_axis, marker="o", hue=hue)
                    elif chart_type in ["bar", "column"]:
                        sns.barplot(data=df, x=x_axis, y=y_axis, hue=hue)
                    elif chart_type == "histogram":
                        plt.hist(df[y_axis], bins=10, alpha=0.7)
                        plt.xlabel(y_axis)
                        plt.ylabel("Frequency")
                    elif chart_type == "scatter":
                        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue)
                    elif chart_type == "pie":
                        plt.pie(df[y_axis], labels=df[x_axis], autopct='%1.1f%%', startangle=90)
                        plt.axis('equal')

                    plt.title(title)
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    os.makedirs("plots", exist_ok=True)
                    filename = f"plots/plot_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()

                    plots_created.append(filename)

                except Exception as e:
                    print(f"Error creating chart {i+1}: {str(e)}")
                    continue

            if plots_created:
                return f"Successfully created {len(plots_created)} plots: {', '.join(plots_created)}"
            else:
                return "No plots could be created from the extracted data."

        except json.JSONDecodeError as e:
            return f"Error parsing LLM response as JSON: {str(e)}"
        except Exception as e:
            return f"Error generating smart plot: {str(e)}"
