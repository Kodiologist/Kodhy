(require [kodhy.macros [block retf]])

(import
  unittest
  [kodhy.util [ret]])

(defn block-f [x]
  (setv l [])
  (.append l 1)
  (.append l (block :foo
    (.append l 2)
    (when (= x 1)
      (.append l 3)
      (ret 4))
    (.append l (block :bar
      (when (= x 2)
        (.append l 5)
        (ret 6))
      (when (= x 3)
        (.append l 7)
        (retf :foo 8))
      9))
    10))
  l)

(defclass C [unittest.TestCase] [

   test-foofy (fn [self]
    (.assertEqual self (block-f 1) [1 2 3 4])
    (.assertEqual self (block-f 2) [1 2 5 6 10])
    (.assertEqual self (block-f 3) [1 2 7 8])
    (.assertEqual self (block-f 4) [1 2 9 10]))])

(when (= __name__ "__main__")
  (setv suite (.loadTestsFromTestCase (unittest.TestLoader) C))
  (.run (unittest.TextTestRunner) suite))
