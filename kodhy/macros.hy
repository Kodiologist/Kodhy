(import
  collections
  [hy [HyList HyDict HyString HySymbol HyExpression]])

(defmacro kwc [f &rest a]
"Keyword call.
    (kwc f 1 :a 3 2 :b 4)  =>  f(1, 2, a = 3, b = 4)
    (kwc f 1 :+a)          =>  f(1, a = True)
    (kwc f 1 :!a)          =>  f(1, a = False)"
  (kwc-f f a))

(defun kwc-f [function input-args]
  (setv input-args (list input-args))
  (setv pargs [])
  (setv kwargs [])
  (while input-args
    (setv x (.pop input-args 0))
    (if (keyword? x) (do
      (setv x (.replace (slice x 2) "-" "_"))
      (.extend kwargs (cond
        [(.startswith x "+")
          [(HyString (slice x 1)) 'True]]
        [(.startswith x "!")
          [(HyString (slice x 1)) 'False]]
        [True
          [(HyString x) (.pop input-args 0)]])))
    ; else
      (.append pargs x)))
  `(apply ~function ~(HyList pargs) ~(HyDict kwargs)))

(defmacro lc [vars a1 &optional a2]
"A more Lispy syntax for list comprehensions.
    (lc [x (range 10)] (str x))
    (lc [x (range 10)] (> x 3) (str x))"
  `(list-comp
    ~(or a2 a1)
    ~vars
    ~@(if a2 [a1] [])))

(defmacro amap [expr args]
"'a' stands for 'anaphoric'."
  `(list (map (lambda [it] ~expr) ~args)))
;  `(list-comp ~expr [it ~args]))  ; The expr doesn't seem to be able to see "it" if it's a (with ...) form.

(defmacro filt [expr args]
  `(list-comp it [it ~args] ~expr))

(defmacro fmap [gen-expr filter-expr args]
  `(list-comp ~gen-expr [it ~args] ~filter-expr))

(defmacro afind [expr args]
  `(try
    (next (filter (lambda [it] ~expr) ~args))
    (catch [_ StopIteration] (raise (ValueError "afind: no matching value found")))))

(defmacro afind-or [expr args &optional [def 'None]]
"The default expression 'def' is evaluated (and its value returned)
if no matching value is found."
  `(try
    (next (filter (lambda [it] ~expr) ~args))
    (catch [_ StopIteration] ~def)))

(defmacro replicate [n &rest body]
  `(list (map (lambda [_] ~@body) (range ~n))))

(defmacro λ [&rest body]
  `(lambda [it] ~@body))

;(defmacro λ2 [&rest body]
;  `(lambda [x y] ~@body))

(defmacro qw [&rest words]
"(qw foo bar baz) => ['foo', 'bar', 'baz']
Caveat: hyphens are transformed to underscores, and *foo* to FOO."
  (HyList (map HyString words)))

(defmacro getl [obj key1 &optional key2 key3]
; Given a pd.DataFrame 'mtcars':
;    (getl mtcars "4 Drive" "hp")    =>  the cell "4 Drive", "hp"
;    (getl mtcars "4 Drive")         =>  the row "4 Drive"
;    (getl mtcars : "hp")            =>  the column "hp"
;    (getl mtcars : (: "cyl" "hp"))  =>  columns "cyl" through "hp"
  (panda-get 'loc obj key1 key2 key3))

(defmacro geti [obj key1 &optional key2 key3]
  (panda-get 'iloc obj key1 key2 key3))

(defmacro $ [obj key]
; Given a pd.DataFrame 'mtcars':
;     ($ mtcars hp)            =>  the column "hp"
  (panda-get 'loc obj : (HyString key)))

(defmacro geta [obj &rest keys]
"For numpy arrays."
  `(get ~obj (, ~@(map parse-key keys))))

(defn parse-key [key]
"Keys can be:
    :  =>  Empty slice object
    (: ...)  =>  slice(...)
    anything else => itself"
  (cond
    [(= key :)
      '((get __builtins__ "slice") None)]
    [(and (instance? HyExpression key) (= (car key) :))
      `((get __builtins__ "slice") ~@(cdr key))]
    [True
      key]))

(defn panda-get [attr obj key1 &optional key2 key3]
  `(get (. ~obj ~attr) ~(cond
    [(is-not key3 None) `(, ~(parse-key key1) ~(parse-key key2) ~(parse-key key3))]
    [(is-not key2 None) `(, ~(parse-key key1) ~(parse-key key2))]
    [True (parse-key key1)])))


(defun dollar-replace [df-sym expr] (cond
  [(isinstance expr HySymbol)
    (if (and (.startswith expr "$") (> (len expr) 1))
      (panda-get 'loc df-sym : (HyString (slice expr 1)))
      expr)]
  [(isinstance expr tuple)
    (tuple (amap (dollar-replace df-sym it) expr))]
  [(and
      (isinstance expr collections.Iterable)
      (not (isinstance expr basestring)))
    (do
      (for [i (range (len expr))]
        (setv (get expr i) (dollar-replace df-sym (get expr i))))
       expr)]
  [True
    expr]))

(defmacro wc [df &rest body]
"With columns.
    (wc df (+ $a $b))  =>  (+ ($ df a) ($ df b))
The replacement is recursive."
  (setv df-sym (gensym))
  (setv body (dollar-replace df-sym body))
  `(let [[~df-sym ~df]] ~@body))

(defmacro ss [df &rest body]
"Subset. Evaluate `body` like `withc`, which should produce a
boolean vector. Return `df` indexed by the boolean vector."
  (setv df-sym (gensym))
  (setv body (dollar-replace df-sym body))
  `(let [[~df-sym ~df]] (get ~df-sym ~@body)))

(defmacro cached [expr &optional [bypass 'None] [cache-dir 'None]]
  `(do
     (import kodhy.util)
     (kodhy.util.cached-eval
       (kodhy.util.show-expr '~expr)
       (fn [] ~expr)
       ~bypass
       ~cache-dir)))
