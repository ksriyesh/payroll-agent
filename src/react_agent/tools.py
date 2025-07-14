"""Tools for the payroll agent."""

from typing import List, Dict, Any
from langchain_core.tools import tool
from react_agent.state import EmployeeData


@tool
def update_state(employees: List[Dict[str, Any]], target_list: str) -> str:
    """Update employee state list."""
    return f"Updated {target_list} with {len(employees)} employees"


def merge_employees(existing: List[EmployeeData], updated: List[EmployeeData]) -> List[EmployeeData]:
    """Automatically merge employee lists with conflict resolution:
    - If employee missing from updated list but exists in existing list: add to updated list
    - If updated employee has empty/zero fields: use existing employee data  
    - Updated employees take precedence over existing
    - All existing employees are preserved in final list
    """
    result = []
    existing_dict = {emp.name: emp for emp in existing}
    updated_dict = {emp.name: emp for emp in updated}
    
    # Process all existing employees first (they all get preserved)
    for existing_emp in existing:
        if existing_emp.name in updated_dict:
            # Employee exists in both lists - merge with updated taking precedence
            updated_emp = updated_dict[existing_emp.name]
            
            # Handle empty fields: use existing data if updated has empty/zero values
            final_regular_hours = updated_emp.regular_hours if updated_emp.regular_hours > 0 else existing_emp.regular_hours
            final_overtime_hours = updated_emp.overtime_hours if updated_emp.overtime_hours > 0 else existing_emp.overtime_hours
            final_payrate = updated_emp.payrate if updated_emp.payrate > 0 else existing_emp.payrate
            
            # Create merged employee with best available data
            merged_emp = EmployeeData(
                name=existing_emp.name,
                regular_hours=final_regular_hours,
                overtime_hours=final_overtime_hours,
                payrate=final_payrate
            )
            result.append(merged_emp)
        else:
            # Employee missing from updated list - add from existing list
            result.append(existing_emp)
    
    # Add any new employees from updated list that don't exist in existing
    for updated_emp in updated:
        if updated_emp.name not in existing_dict:
            result.append(updated_emp)
    
    return result
