(import
  collections)

(defmacro incf [expr]
  `(+= ~expr 1))

(defmacro "#p" [expr]
  ; Pun.
  ; #p a  =>  {"a" a}
  ; #p [a b c]  =>  {"a" a  "b" b  "c" c}
  (when (symbol? expr)
    (setv expr [expr]))
  (hy.models.Dict (+ #* (lfor x expr [(str x) x]))))

(defmacro lc [vars a1 [a2 None]]
"A more Lispy syntax for list comprehensions.
    (lc [x (range 10)] (str x))
    (lc [x (range 10)] (> x 3) (str x))"
  `(lfor
    ~@vars
    ~@(if a2 [:if a1] [])
    ~(or a2 a1)))

(defmacro/g! rmap [arglist #* expr]
; (rmap [[i x] (enumerate "xyzzy")] (setv y (.upper x)) (+ (str i) y))
;   => ["0X" "1Y" "2Z" "3Z" "4Y"]
  (setv [lvalue args] arglist)
  `(list (map
    (fn [~g!arg]
      (setv ~lvalue ~g!arg)
      ~@expr)
    ~args)))

(defmacro amap [expr args]
"'a' stands for 'anaphoric'."
  `(list (map (fn [it] ~expr) ~args)))

(defmacro filt [expr args]
  `(list (filter (fn [it] ~expr) ~args)))

(defmacro fmap [gen-expr filter-expr args]
  `(lfor  it ~args  :if ~filter-expr  ~gen-expr))

(defmacro/g! amap2 [expr args]
; (amap2 (+ a b) (range 10))  =>  [1 5 9 13 17]
  `(do
    (setv ~g!args (list ~args))
    (when (% (len ~g!args) 2)
      (raise (ValueError "iterable argument must have an even number of elements")))
    (list (map
      (fn [~g!i]
        (setv a (get ~g!args ~g!i))
        (setv b (get ~g!args (inc ~g!i)))
        ~expr)
      (range 0 (len ~g!args) 2)))))

(defmacro/g! map-dvals [expr d]
  `(dict (map
    (fn [~g!pair]
      (setv it (get ~g!pair 1))
      (, (get ~g!pair 0) ~expr))
    (.items ~d))))

(defmacro/g! tally [expr args]
  `(do
    (setv ~g!n 0)
    (for [it ~args]
      (when ~expr
        (+= ~g!n 1)))
    ~g!n))

(defmacro afind [expr args]
  `(try
    (next (filter (fn [it] ~expr) ~args))
    (except [StopIteration] (raise (ValueError "afind: no matching value found")))))

(defmacro afind-or [expr args [default None]]
"The default expression 'default' is evaluated (and its value returned)
if no matching value is found."
  `(try
    (next (filter (fn [it] ~expr) ~args))
    (except [StopIteration] ~default)))

(defmacro whenn [expr #* body]
"Analogous to Haskell's liftM for Maybe. Evaluates
'expr' and, if its value is not None, evaluates 'body' with the
value bound to 'it'."
  `(do
    (setv it ~expr)
    (when (is-not it None)
      ~@body)))

(defmacro case [keyform #* clauses]
; (case x [:a 1] [:b 2])
; Implicit progns are provided.
; Returns None if no keys match.
  (case-f keyform clauses None))

(defmacro ecase [keyform #* clauses]
; Like 'case', but throws LookupError if no case matches.
  (case-f keyform clauses
    '[True (raise (LookupError (+ "ecase: No match: " (repr it))))]))

(defn case-f [keyform clauses extra]
  `(do
    (setv it ~keyform)
    (cond
      ~@(gfor form clauses
        `[(= it ~(get form 0))
          ~@(cut form 1 None)])
      ~@(if extra [extra] []))))

(defmacro replicate [n #* body]
  `(list (map (fn [_] ~@body) (range ~n))))

(defmacro block [#* body]
"Evaluate the given expressions while allowing you to jump out
with kodhy.util.ret and kodhy.util.retf. If the first element of
'body' is a keyword, it becomes the name of the block.

The value of the whole expression is that provided by 'ret' or
'retf', if one of those was used, or the last expression otherwise."
  (setv block-name 'None)
  (when (and body (keyword? (get body 0)))
    (setv [block-name body] [(str (get body 0)) (rest body)]))
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

(defmacro retf [block-name [value 'None]]
  (assert (keyword? block-name))
  `(do
    (import [kodhy.util [_KodhyBlockReturn]])
    (raise (_KodhyBlockReturn ~(str block-name) ~value))))

(defn recur-sym-replace [expr f] (cond
  ; Recursive symbol replacement.
  [(isinstance expr hy.models.Symbol)
    (f expr)]
  [(coll? expr)
    ((type expr) (amap (recur-sym-replace it f) expr))]
  [True
    expr]))

(defmacro λ [#* body]
  `(fn [it] ~@body))

;(defmacro λ2 [#* body]
;  `(fn [x y] ~@body))

(defmacro qw [#* words]
"(qw foo bar baz) => ['foo', 'bar', 'baz']"
  (hy.models.List (map hy.models.String words)))

(defmacro meth [param-list #* body]
"(meth [foo] (+ @bar foo))  =>  (fn [self foo] (+ self.bar foo))"
  (meth-f param-list body))

(defmacro cmeth [param-list #* body]
  `(classmethod ~(meth-f param-list body)))

(defn meth-f [param-list body]
  `(fn [self ~@param-list] ~@(recur-sym-replace body (fn [sym] (cond
    [(in sym ["@" "@="])
      sym]
    [(= sym "@@")
      'self]
    [(.startswith sym "@")
      `(. self ~@(amap (hy.models.Symbol it) (.split (cut sym 1 None) ".")))]
    [(.startswith sym "is_@")
      `(. self ~@(amap (hy.models.Symbol it) (.split (+ "is_" (cut sym (len "is_@") None)) ".")))]
    [True
      sym])))))

(defmacro getl [obj key1 [key2 None] [key3 None]]
; Given a pd.DataFrame 'mtcars':
;    (getl mtcars "4 Drive" "hp")    =>  the cell "4 Drive", "hp"
;    (getl mtcars "4 Drive")         =>  the row "4 Drive"
;    (getl mtcars : "hp")            =>  the column "hp"
;    (getl mtcars : (: "cyl" "hp"))  =>  columns "cyl" through "hp"
  (panda-get 'loc obj key1 key2 key3))

(defmacro geti [obj key1 [key2 None] [key3 None]]
  (panda-get 'iloc obj key1 key2 key3))

(setv COLON :)
(defmacro $ [obj key]
; Given a pd.DataFrame 'mtcars':
;     ($ mtcars hp)            =>  the column "hp"
  (panda-get 'loc obj COLON (hy.models.String key)))

(defmacro geta [obj #* keys]
"For numpy arrays."
  `(get ~obj (, ~@(map parse-key keys))))

(defn parse-key [key]
"Keys can be:
    :  =>  Empty slice object
    (: ...)  =>  slice(...)
    anything else => itself"
  (cond
    [(= key :)
      '(slice None)]
    [(and (isinstance key hy.models.Expression) (= (get key 0) :))
      `(slice ~@(cut key 1 None))]
    [True
      key]))

(defn panda-get [attr obj key1 [key2 None] [key3 None]]
  `(get (. ~obj ~attr) ~(cond
    [(is-not key3 None) `(, ~(parse-key key1) ~(parse-key key2) ~(parse-key key3))]
    [(is-not key2 None) `(, ~(parse-key key1) ~(parse-key key2))]
    [True (parse-key key1)])))

(defn dollar-replace [df-sym expr]
  (recur-sym-replace expr (fn [sym]
    (if (.startswith sym "$")
      (if (= (len sym) 1)
        df-sym
        (panda-get 'loc df-sym COLON (hy.models.String (cut sym 1 None))))
      sym))))

(defmacro wc [df #* body]
"With columns.
    (wc df (+ $a $b))  =>  (+ ($ df a) ($ df b))
The replacement is recursive.
`$` on its own becomes simply `df`."
  (setv df-sym (gensym))
  (setv body (dollar-replace df-sym body))
  `(do (setv ~df-sym ~df) ~@body))

(defmacro ss [df #* body]
"Subset. Evaluate `body` like `wc`, which should produce a
boolean vector. Return `df` indexed by the boolean vector."
  (setv df-sym (gensym))
  (setv body (dollar-replace df-sym body))
  `(do (setv ~df-sym ~df) (get ~df-sym ~@body)))

(defmacro ssi [df #* body]
"Subset index. Like `ss`, but returns a list of the indices that
matched."
  (setv df-sym (gensym))
  (setv body (dollar-replace df-sym body))
  `(do
    (setv ~df-sym ~df)
    (.tolist (. (get ~df-sym ~@body) index))))

(defmacro ordf [df #* exprs]
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

(defmacro wcby [df by #* body]
  (setv df-sym (gensym)  it-sym (gensym))
  `(do
    (setv ~df-sym ~df)
    (.apply
      (.groupby ~df-sym ~(dollar-replace df-sym by))
      (fn [~it-sym] ~@(dollar-replace it-sym body)))))

(defmacro/g! cbind [#* args]
 `(do
    (import [kodhy.util [cbind-join :as ~g!cj]])
    (~g!cj "outer" ~@(gfor a args
      (if (keyword? a) a.name a)))))

(defmacro cached [expr [bypass 'None] [cache-dir 'None]]
  `(do
     (import kodhy.util)
     (kodhy.util.cached-eval
       (kodhy.util.show-expr '~expr)
       (fn [] ~expr)
       ~bypass
       ~cache-dir)))

(defmacro show-time-elapsed [#* expr]
  (setv pc (gensym) t (gensym))
  `(do
    (import [time [perf-counter :as ~pc]])
    (setv ~t (~pc))
    (try
      ~@expr
      (finally
        (setv ~t (round (- (~pc) ~t)))
        (print "Time elapsed:" #* (gfor
            [k v] (.items (dict
              :h (// ~t (* 60 60))
              :min (// (% ~t (* 60 60)) 60)
              :s (% ~t 60)))
            :if v
            f"{v} {k}"))))))
