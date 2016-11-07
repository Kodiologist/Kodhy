(import
  collections
  [hy [HyList HyDict HyString HySymbol HyExpression]])

(defn mangle [p]
  ; Mangles a symbol name.
  ; Copied from Hy's parser.py (and translated to Hy).
  (when (and (.startswith p "*") (.endswith p "*") (not-in p ["*" "**"]))
    (setv p (.upper (slice p 1 -1))))
  (unless (= p "-")
    (setv p (.replace p "-" "_")))
  (when (and (.endswith p "?") (!= p "?"))
    (setv p (.format "is_{}" (slice p None -1))))
  p)

(defn implicit-progn [list-of-forms]
  (if (= (len list-of-forms) 1)
    (first list-of-forms)
    `(do ~@list-of-forms)))

(defmacro lc [vars a1 &optional a2]
"A more Lispy syntax for list comprehensions.
    (lc [x (range 10)] (str x))
    (lc [x (range 10)] (> x 3) (str x))"
  `(list-comp
    ~(or a2 a1)
    ~vars
    ~@(if a2 [a1] [])))

(defmacro/g! rmap [arglist &rest expr]
; (rmap [[i x] (enumerate "xyzzy")] (setv y (.upper x)) (+ (string i) y))
;   => ["0X" "1Y" "2Z" "3Z" "4Y"]
  (setv [lvalue args] arglist)
  `(list (map
    (lambda [~g!arg]
      (setv ~lvalue ~g!arg)
      ~@expr)
    ~args)))

(defmacro amap [expr args]
"'a' stands for 'anaphoric'."
  `(list (map (lambda [it] ~expr) ~args)))
;  `(list-comp ~expr [it ~args]))  ; The expr doesn't seem to be able to see "it" if it's a (with ...) form.

(defmacro filt [expr args]
  `(list (filter (lambda [it] ~expr) ~args)))

(defmacro fmap [gen-expr filter-expr args]
  `(list-comp ~gen-expr [it ~args] ~filter-expr))

(defmacro/g! amap2 [expr args]
; (amap2 (+ a b) (range 10))  =>  [1 5 9 13 17]
  `(do
    (setv ~g!args (list ~args))
    (when (% (len ~g!args) 2)
      (raise (ValueError "iterable argument must have an even number of elements")))
    (list (map
      (lambda [~g!i]
        (setv a (get ~g!args ~g!i))
        (setv b (get ~g!args (inc ~g!i)))
        ~expr)
      (range 0 (len ~g!args) 2)))))

(defmacro/g! map-dvals [expr d]
  `(dict (map
    (lambda [~g!pair]
      (setv it (get ~g!pair 1))
      (, (get ~g!pair 0) ~expr))
    (.items ~d))))

(defmacro/g! count [expr args]
  `(do
    (setv ~g!n (int 0))
    (for [it ~args]
      (when ~expr
        (+= ~g!n (int 1))))
    ~g!n))

(defmacro afind [expr args]
  `(try
    (next (filter (lambda [it] ~expr) ~args))
    (except [StopIteration] (raise (ValueError "afind: no matching value found")))))

(defmacro afind-or [expr args &optional [def 'None]]
"The default expression 'def' is evaluated (and its value returned)
if no matching value is found."
  `(try
    (next (filter (lambda [it] ~expr) ~args))
    (except [StopIteration] ~def)))

(defmacro whenn [expr &rest body]
"Analogous to Haskell's liftM for Maybe. Evaluates
'expr' and, if its value is not None, evaluates 'body' with the
value bound to 'it'."
  `(do
    (setv it ~expr)
    (when (is-not it None)
      ~@body)))

(defmacro case [keyform &rest clauses]
; (case x [:a 1] [:b 2])
; Implicit progns are provided.
; Returns None if no keys match.
  (case-f keyform clauses None))

(defmacro ecase [keyform &rest clauses]
; Like 'case', but throws LookupError if no case matches.
  (case-f keyform clauses
    '[True (raise (LookupError (+ "ecase: No match: " (repr it))))]))

(defn case-f [keyform clauses extra]
  `(do
    (setv it ~keyform)
    (cond
      ~@(lc [form clauses]
        `[(= it ~(first form))
          ~@(slice form 1)])
      ~@(if extra [extra] []))))

(defmacro replicate [n &rest body]
  `(list (map (lambda [_] ~@body) (range ~n))))

(defmacro block [&rest body]
"Evaluate the given expressions while allowing you to jump out
with kodhy.util.ret and kodhy.util.retf. If the first element of
'body' is a keyword, it becomes the name of the block.

The value of the whole expression is that provided by 'ret' or
'retf', if one of those was used, or the last expression otherwise."
  (setv block-name 'None)
  (when (and body (keyword? (first body)))
    (setv [block-name body] [(first body) (rest body)]))
  (setv r (gensym))
  `(do (import [kodhy.util [_KodhyBlockReturn]]) (try
    (do ~@body)
    (except [~r _KodhyBlockReturn]
      (if (and (. ~r block-name) (!= (. ~r block-name) ~block-name))
        ; If the return named a block, and its name doesn't
        ; match ours, keep bubbling upwards.
        (raise)
        ; Otherwise, we can stop here. Return the return value.
        (. ~r value))))))

(defn recur-sym-replace [expr f] (cond
  ; Recursive symbol replacement.
  [(instance? HySymbol expr)
    (f expr)]
  [(instance? tuple expr)
    (tuple (amap (recur-sym-replace it f) expr))]
  [(coll? expr)
    (for [i (range (len expr))]
      (setv (get expr i) (recur-sym-replace (get expr i) f)))
     expr]
  [True
    expr]))

(defmacro λ [&rest body]
  `(lambda [it] ~@body))

;(defmacro λ2 [&rest body]
;  `(lambda [x y] ~@body))

(defmacro qw [&rest words]
"(qw foo bar baz) => ['foo', 'bar', 'baz']
Caveat: hyphens are transformed to underscores, and *foo* to FOO."
  (HyList (map HyString words)))

(defmacro meth [param-list &rest body]
  `(fn [self ~@param-list] ~@(recur-sym-replace body (fn [sym] (cond
    [(.startswith sym "@")
      (if (= sym "@")
        'self
        `(. self ~@(amap (HySymbol it) (.split (slice sym 1) "."))))]
    [(.startswith sym "is_@")
      `(. self ~@(amap (HySymbol it) (.split (+ "is_" (slice sym (len "is_@"))) ".")))]
    [True
      sym])))))

(defmacro cmeth [param-list &rest body]
  `(classmethod (meth ~param-list ~@body)))

(defmacro defcls [name inherit &rest body]
  `(defclass ~name ~inherit ~@(amap2
    (HyList [a b])
    body)))

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

(defn dollar-replace [df-sym expr]
  (recur-sym-replace expr (fn [sym]
    (if (.startswith sym "$")
      (if (= (len sym) 1)
        df-sym
        (panda-get 'loc df-sym : (HyString (slice sym 1))))
      sym))))

(defmacro wc [df &rest body]
"With columns.
    (wc df (+ $a $b))  =>  (+ ($ df a) ($ df b))
The replacement is recursive.
`$` on its own becomes simply `df`."
  (setv df-sym (gensym))
  (setv body (dollar-replace df-sym body))
  `(do (setv ~df-sym ~df) ~@body))

(defmacro ss [df &rest body]
"Subset. Evaluate `body` like `wc`, which should produce a
boolean vector. Return `df` indexed by the boolean vector."
  (setv df-sym (gensym))
  (setv body (dollar-replace df-sym body))
  `(do (setv ~df-sym ~df) (get ~df-sym ~@body)))

(defmacro ssi [df &rest body]
"Subset index. Like `ss`, but returns a list of the indices that
matched."
  (setv df-sym (gensym))
  (setv body (dollar-replace df-sym body))
  `(do
    (setv ~df-sym ~df)
    (.tolist (. (get ~df-sym ~@body) index))))

(defmacro ordf [df &rest exprs]
"Order data frame. (ordf d (.abs $baz) $bar) sorts first by the
absolute value of the column `baz`, then by `bar`."
  (setv [df-sym pd sorting-df] [(gensym) (gensym) (gensym)])
  (setv exprs (dollar-replace df-sym exprs))
  `(do
    (setv ~df-sym ~df)
    (import [pandas :as ~pd])
    (setv ~sorting-df (.reset-index (.concat ~pd [~@exprs] 1) None True))
    (geti ~df-sym (. (.sort-values ~sorting-df (list (. ~sorting-df columns))) index))))
; ~pd

(defmacro/g! cbind [&rest args]
 `(do
    (import [kodhy.util [cbind-join :as ~g!cj]])
    (~g!cj "outer" ~@(lc [a args]
      (if (keyword? a) (name a) a)))))

(defmacro cached [expr &optional [bypass 'None] [cache-dir 'None]]
  `(do
     (import kodhy.util)
     (kodhy.util.cached-eval
       (kodhy.util.show-expr '~expr)
       (fn [] ~expr)
       ~bypass
       ~cache-dir)))
