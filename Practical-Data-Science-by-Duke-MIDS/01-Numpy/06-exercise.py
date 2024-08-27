import numpy as np


# Ques_1. Create the following vector of salaries: 50_000, 105_250, 55_000, 89_000. What is the total payroll (sum of all salaries for the company)?
salary = np.array([50_000, 105_250, 55_000, 89_000])
total_payroll1 = np.sum(salary)
print(total_payroll1)

# Ques_2. Now suppose our evil CEO has decided to give herself a raise. Take your salary vector and modify it so that the CEO – the person making 105,250 dollars – gets a raise of 15%.
ceo_salary = salary[salary ==  105_250]
adjusted_ceo_salary = ceo_salary * 1.15
print(adjusted_ceo_salary)
salary[salary ==  105_250] = adjusted_ceo_salary
total_payroll2 = np.sum(salary)
print(total_payroll2)
# salary[salary == 105_250] = salary[salary == 105_250] * 1.15

# Ques_3. 115% of 105,250 dollars is 121,037.50 dollars. Is that the value in your array? If not, can you tell why not?


# Ques_4. Recreate your vector, and do something with the dtype argument so that when you give the CEO a raise of 15%, she ends up with a salary of 121,037.50 dollars.
salary = np.array([50_000, 105_250, 55_000, 89_000], dtype="float")
salary[salary == 105_250] = salary[salary == 105_250] * 1.15
print(salary)

# Ques_5. Now suppose this has so annoyed the lowest paid employee (the woman earning 50,000 dollars) that she demands a raise as well. Increase her salary by 20%.
lowest_salary = salary[salary ==  50_000]
adjusted_lowest_salary = lowest_salary * 1.20
print(adjusted_lowest_salary)
salary[salary ==  50_000] = adjusted_lowest_salary
print(salary)

# Ques_6. This has so irritated the other two employees you must now give them 10% raises. Increase their salaries by 10%.
rest_salary = salary[(salary ==  55_000) | (salary == 89_000)]
rest_salary_increase = rest_salary * 1.10
salary[(salary ==  55_000) | (salary == 89_000)] = rest_salary_increase
final_payroll = np.sum(salary)
print(salary)

# Ques_7. Now calculate the total payroll for the company. In the end, what did the CEO’s ~16,000 raise end up costing the company?
added_cost = final_payroll - total_payroll1
print(f"The MD/CEO's ~16,000 salary increase ended up causing a total of {added_cost:.0f} in increment to the payroll.")

