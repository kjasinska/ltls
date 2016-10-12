# ltls

##Requirements:
    - multi-label datasets must have sorted labels lists for each instance - sklearn libsvm reader requirement

##Running:
    - to repeat experiments from the paper:
        python make.py [data set name]
		- use data sets from [link](http://www.cs.utexas.edu/~xrhuang/PDSparse/)
        - data set files must be in data/[data set name]/, named [data set name].{train/test/heldout}
        - will use parameters specified in make.py params variable
        - model will be saved to models/[data set name]/
        - log will be saved to logs/[data set name]

    - to use to your own experiments:
        You can use make_example.py as a template for your own experiments

