#!/usr/bin/env python3
"""
IEEE Standard System Power Flow Analysis
Main program to run power flow on IEEE test systems
"""

from PowerFlowSolver import PowerFlowSolver
import sys

# Available IEEE test systems
IEEE_SYSTEMS = {
    '9': 'IEEE 9-bus system',
    '14': 'IEEE 14-bus system', 
    '30': 'IEEE 30-bus system',
    '57': 'IEEE 57-bus system',
    '118': 'IEEE 118-bus system'
}

def display_menu():
    """Display available IEEE test systems"""
    print("\n" + "="*50)
    print("IEEE Standard System Power Flow Analysis")
    print("="*50)
    print("Available systems:")
    for key, description in IEEE_SYSTEMS.items():
        print(f"  {key}: {description}")
    print("  q: Quit")
    print("-"*50)

def get_user_choice():
    """Get and validate user input"""
    while True:
        choice = input("Select system (9/14/30/57/118) or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            return None
        elif choice in IEEE_SYSTEMS:
            return choice
        else:
            print("Invalid choice! Please select from available options.")

def run_power_flow(system_choice):
    """Run power flow for selected IEEE system"""
    try:
        print(f"\nRunning power flow for {IEEE_SYSTEMS[system_choice]}...")
        
        # Initialize solver with selected system
        solver = PowerFlowSolver(ieee_system=int(system_choice))
        
        # Run power flow
        results = solver.run_power_flow()
        
        print(f"✓ Power flow completed successfully!")
        print(f"✓ System: {IEEE_SYSTEMS[system_choice]}")
        
        return results
        
    except Exception as e:
        print(f"✗ Error running power flow: {str(e)}")
        return None

def main():
    """Main program execution"""
    print("Starting IEEE Power Flow Analysis Tool...")
    
    while True:
        display_menu()
        choice = get_user_choice()
        
        if choice is None:  # User chose to quit
            print("\nExiting program. Goodbye!")
            break
            
        # Run power flow for selected system
        results = run_power_flow(choice)
        
        if results is not None:
            # Ask if user wants to run another system
            continue_choice = input("\nRun another system? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\nExiting program. Goodbye!")
                break

if __name__ == "__main__":
    main()
