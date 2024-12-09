#importing core libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as stats
import statsmodels.api as sm
import scikit_posthocs as sp
#importing specific statistical modeling and diagnostics tools
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from scipy.stats import mannwhitneyu, f_oneway, kruskal
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.decomposition import PCA
#loading file
file_path = "C:/Users/Python/Downloads/archive (1)/Wellbeing_and_lifestyle_data_Kaggle.csv"
df = pd.read_csv(file_path)
#making sure that the file has been loaded/read correctly and exploring data
print(df.head())
print(df.info())
print(df.describe())
#checking for missing values 
print(df.isnull().sum())
#no need to drop missing values here, since there are no NA values
# exploring unique values in each column
for col in df.columns:
    unique_values = df[col].unique()
    print(f"\nUnique values in {col}: {len(unique_values)} unique values")
    if len(unique_values) < 10:
        print(unique_values)
    else:
        print(f"Showing first 5 unique values: {unique_values[:5]} ...")
#checking for duplicates 
print(df.duplicated().sum())
#no duplicates found
#no need to include Timestamp for further statistical analysis
df = df.drop("Timestamp",axis=1)

#transforming age categories into ordinal category for further analysis
#reason for this transformation is to enable checking correlation between work-life-balance-score and ordinal age categories
#represents true nature of age categories (being ordinal) better than string-based categories alone
#treating age as continuous by substituting age categories with mean age of each range might alter nature of data since we do not know what the age distribution really looks like
age_order = ["Less than 20","21 to 35","36 to 50", "51 or more"]
encoder = OrdinalEncoder(categories=[age_order]) 
df['AGE'] = encoder.fit_transform(df[['AGE']])
df['AGE'] = df['AGE'] + 1
#visualizing transformed age column 
sns.countplot(x='AGE', data=df)
plt.title('Age Categories')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.show()

#transforming gender into binary values
df = pd.get_dummies(df, columns= ["GENDER"],drop_first = True)
# Replace invalid entries with median
df['DAILY_STRESS'] = pd.to_numeric(df['DAILY_STRESS'], errors='coerce')
median_stress = df['DAILY_STRESS'].median()
df['DAILY_STRESS'].fillna(median_stress, inplace=True)

# checking for outliers in lifestyle factors
df_without_work_life_balance = df.drop(columns=['WORK_LIFE_BALANCE_SCORE'])
z_scores = stats.zscore(df_without_work_life_balance.select_dtypes(include=[np.number]))
outliers = np.abs(z_scores) > 3
#visually checking for outliers in lifestyle factors
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_without_work_life_balance.select_dtypes(include=[np.number]))
plt.tight_layout()
plt.show()
#outliers in daily shouting and sleep can be seen
# but seem probable and do not have to be dropped, since they likely represent real data

#general questions worth investigating:
# 1. what are the correlations between the  reported lifestyle factors and wellbeing scores? 
# 2. can we predict work-life-balance-scores based on lifestyle factors using a predictive model?
# 3. are there any significant differences in wellbeing scores based on age or gender? 
# can differences for the most important lifestyle factors based on gender be found?

#computing correlation matrix to explore first research question
correlation_matrix = df.corr()
#visualizing the correlation matrix
plt.figure(figsize=(22, 2.5))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix", size=14)
plt.subplots_adjust(hspace=0.2)
plt.tight_layout()
plt.show()
#highest positive correlations between wellbeing scores and: achievement, supporting others and todo completed
# highest negative correlations between wellbeing scores and: daily stress, lost vacation and daily shouting 

#visualizing distributions of the lifestyle factors measured and the actual wellbeing scores
df.hist(bins=20, figsize=(17, 15))
plt.suptitle("histograms of lifestyle factors and wellbeing score", size=14)
plt.show()
# going on to test for normality in the different columns
for col in df.columns:
    stat, p = stats.shapiro(df[col])
    if p < 0.05:
        print(f"{col} is not normally distributed (p={p:.3f})")
    else:
        print(f"{col} is normally distributed (p={p:.3f})")
# no normal distributions can be found 

#proceeding to investigate correlations mentioned above (found through correlation matrix)
#achievement and work life balance score

def correlation_analysis(x, y, x_label, y_label, plot_title):
    corr, p_value = stats.spearmanr(x, y)
    print(f"Spearman correlation: {corr}")
    print(f"P-value: {p_value}")
    
    # plotting correlation:
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.show() 

correlation_analysis(df['ACHIEVEMENT'], df['WORK_LIFE_BALANCE_SCORE'], 
                                  'Achievement', 'Work-Life-Balance-Score', 
                                  'Achievement vs. Work-Life-Balance-Score')
# p-value below 0.05

correlation_analysis(df['SUPPORTING_OTHERS'], df['WORK_LIFE_BALANCE_SCORE'], 
                                  'Supporting Others', 'Work-Life-Balance-Score', 
                                  'Supporting Others vs. Work-Life-Balance-Score')
#p-value below 0.05

correlation_analysis(df['TODO_COMPLETED'], df['WORK_LIFE_BALANCE_SCORE'], 
                                  'ToDo Completed', 'Work-Life Balance-Score', 
                                  'ToDo Completed vs. Work-Life Balance-Score')
#p-value below 0.05

correlation_analysis(df['DAILY_STRESS'], df['WORK_LIFE_BALANCE_SCORE'], 
                                  'Daily Stress', 'Work-Life Balance Score', 
                                  'Daily Stress vs. Work-Life Balance Score')
#p-value below 0.05

correlation_analysis(df['LOST_VACATION'], df['WORK_LIFE_BALANCE_SCORE'], 
                                  'Lost Vacation', 'Work-Life Balance Score', 
                                  'Lost Vacation vs. Work-Life Balance Score')
#p-value above 0.05, no statistical significance found

correlation_analysis(df['DAILY_SHOUTING'], df['WORK_LIFE_BALANCE_SCORE'], 
                                  'Daily Shouting', 'Work-Life Balance Score', 
                                  'Daily Shouting vs. Work-Life Balance Score')
#p-value above 0.05, no statistical significance found
#4 out of the 6 factors carry statistic significance and will be investigated further

#regression analysis to check for causality after finding significance through p-values for all significant factors
X = df[['ACHIEVEMENT', 'SUPPORTING_OTHERS', 'TODO_COMPLETED', "DAILY_STRESS"]]
X = sm.add_constant(X)  
y = df['WORK_LIFE_BALANCE_SCORE']

model = sm.OLS(y, X).fit()
residuals = model.resid
fitted = model.fittedvalues
print(model.summary())

#visualizing residuals vs. fitted
plt.figure(figsize=(8, 5))
sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={"color": "blue"})
plt.title("Residuals vs. Fitted")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()

#q-q plot
sm.qqplot(residuals, line='s')
plt.title("Q-Q Plot")
plt.show()

_, pval, _, _ = het_breuschpagan(residuals, X)
print(f"Breusch-Pagan p-value: {pval}")
#Breusch-Pagan p-value indicates heteroscedasticity, violating assumptions needed for OLS 

#using robust standard errors instead
x_with_const = sm.add_constant(X) 
model = sm.OLS(y, x_with_const).fit(cov_type='HC3')
print(model.summary())

#rerunning Breusch-Pagan to see if high heteroscedasticity remains
_, pval, _, _ = het_breuschpagan(residuals, X)
print(f"Breusch-Pagan p-value: {pval}")

#Breusch-Pagan p-value still indicates high heteroscedasticity
# this violates the assumptions needed for OLS modelling
#WLS might be better approach for modelling, since it allows for varying levels of variance by assigning weights
weights = 1 / (model.resid ** 2)  
model = sm.WLS(y, x_with_const, weights=weights).fit()
print(model.summary())
#visualizing residuals vs. fitted for the new WLS model
residuals = model.resid
plt.figure(figsize=(8, 5))
sns.residplot(x=model.fittedvalues, y=residuals, lowess=True, line_kws={"color": "red"})
plt.title("Residuals vs. Fitted for WLS Model")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()
#visualozing distribution of weights
plt.hist(weights, bins=30, edgecolor='k')
plt.title("Distribution of Weights")
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.show()
#evaluating model by using cross-validation to estimate out of sample performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_cv = sm.WLS(y_train, X_train, weights=weights.loc[X_train.index]).fit()
y_pred = model_cv.predict(X_test)

r2_cv = r2_score(y_test, y_pred)
print("Cross-Validated R-squared:", r2_cv)
#generating robust standard errors
model_robust = model_cv.get_robustcov_results(cov_type='HC3')
print(model_robust.summary())

aic = model.aic
bic = model.bic
adj_r2 = model.rsquared_adj

print(f"AIC: {aic}")
print(f"BIC: {bic}")
print(f"Adjusted R-squared: {adj_r2}\n")
#AIC: 130796.58806998274, BIC: 130834.98103232366, Adjusted R-squared: 0.9999650963500583 -> model might be a bit too complex
# high adjusted r-squared might indicate overfitting or good model performance

#using LassoCV below for cross-validation 
#trying to avoid overfitting at this point (considering adj_r2)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(alphas=np.logspace(-6, 6, 13), max_iter=10000)
lasso.fit(X_scaled, y)

print(f"best alpha for Lasso: {lasso.alpha_}")
print(f"Lasso Model R^2: {lasso.score(X_scaled, y)}")
print(f"Lasso Coefficients: {lasso.coef_}") 

#Lasso Model suggests best alpha to be 0.001 and and assigned achievement a coefficient of 0
#indicates low significance of achievement for explaining variance of wellbeing score
#overfitting seems well-controlled with achievement having been erased from model
#  63 % of variance in work-life-balance score can be explained by model 

# stress has largest impact overall on score (albeit negative), biggest positive impact has todos completed

# checking for multicollinearity, leaving out achievement since Lasso Model suggested it to be "meaningless"

X = df[['SUPPORTING_OTHERS', 'TODO_COMPLETED','DAILY_STRESS']]
X = sm.add_constant(X)
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif)

#VIF scores for individual factors all below 5, for constant term 13.5 (can be explained by baseline wellbeing score)
#concerns of multicollinearity are not necessary 

#moving onto looking at the effect of gender
#inspecting scores grouped by gender
gender_grouped = df.groupby('GENDER_Male')[['WORK_LIFE_BALANCE_SCORE', 'ACHIEVEMENT', 'SUPPORTING_OTHERS', "TODO_COMPLETED", "DAILY_STRESS"]].mean()
print(gender_grouped)

#males seem to score slightly lower for the work-life-balance score, supporting others, daily stress and todo completed
# males seem to score slightly higher in achievemet 
#checking for significance 
variables= ['WORK_LIFE_BALANCE_SCORE', 'ACHIEVEMENT', 'SUPPORTING_OTHERS', "TODO_COMPLETED", "DAILY_STRESS"]
results_gender=[]
for var in variables:
    males= df[df["GENDER_Male"]== True][var]
    females = df[df["GENDER_Male"]== False] [var]
    u_stat, p_value=mannwhitneyu(males, females, alternative = "two-sided")
    results_gender.append({
        "Variable":var,
        "U-Statistic":u_stat,
        "P-Value":p_value 
    })
print(results_gender)
#significant group differences across genders found for work-life-balance-score, supporting others, todo completed and daily stress
# no significant group difference across gender regarding achievement 
#Bonferroni adjustment, leaving out insignificant achievement
results_gender = [
    {'Variable': 'WORK_LIFE_BALANCE_SCORE', 'U-Statistic': 28586565.0, 'P-Value': 4.9466747386063043e-08},
    {'Variable': 'SUPPORTING_OTHERS', 'U-Statistic': 25422139.5, 'P-Value': 1.9695421649158092e-63},
    {'Variable': 'TODO_COMPLETED', 'U-Statistic': 27319502.0, 'P-Value': 1.568152080232235e-23},
    {'Variable': 'DAILY_STRESS', 'U-Statistic': 25975369.0, 'P-Value': 7.53074228281954e-51}
]
num_tests = len(results_gender)
for result_gender in results_gender:
    result_gender['Bonferroni_P-Value'] = min(result_gender['P-Value'] * num_tests, 1.0)  

for result_gender in results_gender:
    print(f"Variable: {result_gender['Variable']}, Original P-Value: {result_gender['P-Value']:.2e}, Bonferroni-Adjusted P-Value: {result_gender['Bonferroni_P-Value']:.2e}")

#adjusted p-values for work_life_balance score, supporting others, todo completed and daily stress remain significant (< 0.05)

#investigating effect of age on work-life-balance-score
# since no normality was found ANOVA cannot be used

groups = [df[df['AGE'] == i]['WORK_LIFE_BALANCE_SCORE'].values for i in df['AGE'].unique()]
stat, p_value= kruskal(*groups)
print(f'Kruskal-Wallis test: statistic = {stat}, p-value = {p_value}')
#Kruskal-Wallis results indicate that differences across groups are significant, post-hoc testing:
posthoc = sp.posthoc_dunn(df, val_col='WORK_LIFE_BALANCE_SCORE', group_col='AGE', p_adjust='bonferroni')
print(posthoc)
#significant wellbeing score differences across  when comparing groups 1&2, 1&4, 2&3
#looking at mean scores

age_group_means = df.groupby("AGE")["WORK_LIFE_BALANCE_SCORE"].mean()
print(age_group_means)
#age category 1 reports higher scores than group 2, but lower than the oldest age group 
# age category 3 has significantly higher wellbeing scores

#conclusion:
#supporting others and todo completed seem to have the biggest positive effect on wellbeing scores
#daily stress has the biggest negative effect
#we can predict well-being scores based on these factors to a certain extent (63% of variance explained)
#significant group differences across genders found for work-life-balance-score, supporting others, todo completed and daily stress
# no significant group difference across gender regarding achievement 
#significant wellbeing score differences across  when comparing groups 1&2, 1&4, 2&3
# it seems that the oldest employees experience the highest work-life-balance

