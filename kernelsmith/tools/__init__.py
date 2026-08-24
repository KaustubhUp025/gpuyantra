"""ADK FunctionTools, one module per tool.

Nothing is re-exported here on purpose: each module already exposes a `*_tool` object
whose name matches the module (`retrieval_tool.retrieval_tool`), and binding those at
package level would shadow the modules themselves. Import from the module:

    from kernelsmith.tools.retrieval_tool import retrieval_tool

Each module pairs a plain Python function — what the tests and scripts call — with the
FunctionTool wrapper the agents get. Nothing here executes generated code:
`verifier_tool` shells out to the sandbox subprocess (red line #2).
"""
