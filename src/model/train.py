from rpy2.robjects import r


def main():
    r.install.packages("caret", repos="http://cran.us.r-project.org")
    r.install.packages("e1071", repos="http://cran.us.r-project.org")
    r.source("train.py")


if __name__ == "__main__":
    main()
