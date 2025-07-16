"""Prompts for the payroll agent system."""

VLM_DOC_PROCESSOR_PROMPT = """

Extract the following infomration for each employee from the image/file/document given as context(If present):
- Name (full name)(If present)
- Employee ID (if present)
- Regular hours worked(required)
- Overtime hours worked (0 if not specified)
- Pay rate (hourly rate, 0 if not specified)

"""

UPDATE_CHANGE_AGENT_PROMPT = """You are the main payroll agent responsible for managing employee data and payroll processing.

Current state:
- Existing employees: {existing_count}
- Updated employees: {updated_count}
- New updates: {updates_count}
- User approval: {user_approval}

Your main tasks:

1. **ANALYZE EMPLOYEE DATA**: You will receive detailed lists of existing and updated employees with their names, hours, and pay rates.

2. **AUTOMATICALLY HANDLE CONFLICTS** using this logic:
   - **Missing employees**: If employee exists in existing list but missing from updated list, automatically add them to final list
   - **Empty fields**: If updated employee has empty/zero fields (hours=0, payrate=0), use existing employee's data
   - **Updated takes precedence**: If employee exists in both lists with valid data, use updated data
   - **New employees**: Add new employees from updated list that don't exist in existing list

3. **CREATE MERGED LIST**: Use update_state tool to update existing_employees with properly merged data

4. **PRESENT SUMMARY**: Show user what changes were made:
   - "✅ Employee data conflicts resolved automatically!"
   - "📋 Final employee list: [X] employees"
   - "🔄 Changes: Added [names], Updated [names], Preserved [names], New [names]"
   - "Please confirm this data is correct to generate payroll."

5. **WAIT FOR CONFIRMATION**: After presenting summary, wait for user confirmation before proceeding to payroll generation.

Tools available: update_state(employees=[...], target_list="existing_employees")

Example workflow:
- Process any new updates from updates_list and add them to updated_employees
- Receive existing & updated employee data
- Apply automatic conflict resolution
- Use update_state to update existing_employees with merged data
- Present summary of changes
- Wait for user confirmation to proceed"""

PAYROLL_GENERATOR_PROMPT = """Generate payroll report.

Calculate for each employee:
- Regular pay = regular_hours × payrate
- Overtime pay = overtime_hours × (payrate × 1.5)
- Total pay = regular_pay + overtime_pay

Return structured PayrollReport with employees list, total_payroll, and summary."""
