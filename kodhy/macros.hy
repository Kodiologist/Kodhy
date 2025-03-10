(require
  hyrule [defmacro!])

(import
  collections
  hyrule [coll?])

(defmacro incf [expr]
  `(+= ~expr 1))

(defreader p
  ; Pun.
  ; #p a  =>  {"a" a}
  ; #p [a b c]  =>  {"a" a  "b" b  "c" c}
  (setv expr (.parse-one-form &reader))
  (when (isinstance expr hy.models.Symbol)
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

(defmacro! rmap [arglist #* expr]
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

(defmacro! amap2 [expr args]
; (amap2 (+ a b) (range 10))  =>  [1 5 9 13 17]
  `(do
    (setv ~g!args (list ~args))
    (when (% (len ~g!args) 2)
      (raise (ValueError "iterable argument must have an even number of elements")))
    (list (map
      (fn [~g!i]
        (setv a (get ~g!args ~g!i))
        (setv b (get ~g!args (+ ~g!i 1)))
        ~expr)
      (range 0 (len ~g!args) 2)))))

(defmacro! map-dvals [expr d]
  `(dict (map
    (fn [~g!pair]
      (setv it (get ~g!pair 1))
      #((get ~g!pair 0) ~expr))
    (.items ~d))))

(defmacro! tally [expr args]
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

(defmacro replicate [n #* body]
  `(list (map (fn [_] ~@body) (range ~n))))

(defn recur-sym-replace [expr f] (cond
  ; Recursive symbol replacement.
  (isinstance expr hy.models.Symbol)
    (f expr)
  (coll? expr)
    ((type expr) (amap (recur-sym-replace it f) expr))
  True
    expr))

(defmacro λ [#* body]
  `(fn [it] ~@body))

;(defmacro λ2 [#* body]
;  `(fn [x y] ~@body))

(defmacro qw [#* words]
"(qw foo bar baz \"a b\" 5) => ['foo', 'bar', 'baz', 'a b', '5']"
  (hy.models.List (gfor  w words  (hy.models.String
    (if (isinstance w #(hy.models.Symbol hy.models.String))
      w
      (.lstrip (hy.repr w) "'"))))))

(defmacro meth [param-list #* body]
"(meth [foo] (+ @bar foo))  =>  (fn [self foo] (+ self.bar foo))"
  (meth-f param-list body))

(defmacro cmeth [param-list #* body]
  `(classmethod ~(meth-f param-list body)))

(defn meth-f [param-list body]
  `(fn [self ~@param-list] ~@(recur-sym-replace body (fn [sym] (cond
    (in sym ["@" "@="])
      sym
    (= sym "@@")
      'self
    (.startswith sym "@")
      `(. self ~@(amap (hy.models.Symbol it) (.split (cut sym 1 None) ".")))
    (.startswith sym "is_@")
      `(. self ~@(amap (hy.models.Symbol it) (.split (+ "is_" (cut sym (len "is_@") None)) ".")))
    True
      sym)))))

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
  `(get ~obj ~(hy.models.String key)))

(defmacro geta [obj #* keys]
"For numpy arrays."
  `(get ~obj #(~@(map parse-key keys))))

(defn parse-key [key]
"Keys can be:
    :  =>  Empty slice object
    (: ...)  =>  slice(...)
    anything else => itself"
  (cond
    (= key :)
      '(slice None)
    (and (isinstance key hy.models.Expression) (= (get key 0) :))
      `(slice ~@(cut key 1 None))
    True
      key))

(defn panda-get [attr obj key1 [key2 None] [key3 None]]
  `(get (. ~obj ~attr) ~(cond
    (is-not key3 None) `#(~(parse-key key1) ~(parse-key key2) ~(parse-key key3))
    (is-not key2 None) `#(~(parse-key key1) ~(parse-key key2))
    True (parse-key key1))))

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
  (setv df-sym (hy.gensym))
  (setv body (dollar-replace df-sym body))
  `(do (setv ~df-sym ~df) ~@body))

(defmacro ss [df #* body]
"Subset. Evaluate `body` like `wc`, which should produce a
boolean vector. Return `df` indexed by the boolean vector."
  (setv df-sym (hy.gensym))
  (setv body (dollar-replace df-sym body))
  `(do (setv ~df-sym ~df) (get ~df-sym ~@body)))

(defmacro ssi [df #* body]
"Subset index. Like `ss`, but returns a list of the indices that
matched."
  (setv df-sym (hy.gensym))
  (setv body (dollar-replace df-sym body))
  `(do
    (setv ~df-sym ~df)
    (.tolist (. (get ~df-sym ~@body) index))))

(defmacro ordf [df #* exprs]
"Order data frame. (ordf d (.abs $baz) $bar) sorts first by the
absolute value of the column `baz`, then by `bar`."
  (setv [df-sym pd sorting-df] [(hy.gensym) (hy.gensym) (hy.gensym)])
  (setv exprs (dollar-replace df-sym exprs))
  `(do
    (setv ~df-sym ~df)
    (import  pandas :as ~pd)
    (setv ~sorting-df (.reset-index (.concat ~pd [~@exprs] 1) None True))
    (geti ~df-sym (. (.sort-values ~sorting-df (list (. ~sorting-df columns))) index))))
; ~pd

(defmacro wcby [df by #* body]
  (setv df-sym (hy.gensym)  it-sym (hy.gensym))
  `(do
    (setv ~df-sym ~df)
    (.apply
      (.groupby ~df-sym ~(dollar-replace df-sym by))
      (fn [~it-sym] ~@(dollar-replace it-sym body)))))

(defmacro! cbind [#* args]
 `(do
    (import  kodhy.util [cbind-join :as ~g!cj])
    (~g!cj "outer" ~@(gfor a args
      (if (isinstance a hy.models.Keyword) a.name a)))))

(defmacro cached [expr [bypass 'None] [cache-dir 'None]]
  `(do
     (import kodhy.util)
     (kodhy.util.cached-eval
       (kodhy.util.show-expr '~expr)
       (fn [] ~expr)
       ~bypass
       ~cache-dir)))

(defmacro show-time-elapsed [#* expr]
  (setv t (hy.gensym))
  `(do
    (setv ~t (hy.I.time.perf-counter))
    (try
      ~@expr
      (finally
        (setv ~t (round (- (hy.I.time.perf-counter) ~t)))
        (print "Time elapsed:" #* (gfor
            [k v] (.items (dict
              :h (// ~t (* 60 60))
              :min (// (% ~t (* 60 60)) 60)
              :s (% ~t 60)))
            :if v
            f"{v} {k}"))))))
