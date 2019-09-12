(import
  [pandas :as pd]
  [kodhy.util [recategorize]])

(defn test-recat []
  (setv x (pd.Series (list "aabaddabcabbdbb") :dtype "category"))
  (assert (= (.join "" x.cat.categories) "abcd"))
  (setv x (recategorize x
    "a" "Z"
    "b" "Z"
    "d" None
    "c" "X"))
  (assert (= (.join "" x.cat.categories) "ZX"))
  (assert (= (.join "-" (map str x)) "Z-Z-Z-Z-nan-nan-Z-Z-X-Z-Z-Z-nan-Z-Z")))
