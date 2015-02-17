(require kodhy.macros)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Numbers and arrays
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn logit [x]
  (import numpy)
  (numpy.log (/ x (- 1 x))))

(defn ilogit [x]
  (import numpy)
  (/ 1 (+ 1 (numpy.exp (- x)))))

(defn zscore [x]
  (/ (- x (.mean x)) (kwc .std x :ddof 0)))

(defn hzscore [x]
"Half the z-score. Divides by two SDs instead of one, per:
Gelman, A. (2008). Scaling regression inputs by dividing by two standard deviations. Statistics in Medicine, 27(15), 2865–2873. doi:10.1002/sim.3107"
  (/ (- x (.mean x)) (* 2 (kwc .std x :ddof 0))))

(defn rmse [v1 v2]
"Root mean square error."
  (import [numpy :as np])
  (np.sqrt (np.mean (** (- v1 v2) 2))))

(defn mean-ad [v1 v2]
"Mean absolute deviation."
  (import [numpy :as np])
  (np.mean (np.abs (- v1 v2))))

(defn valcounts [x]
  (import [pandas :as pd])
;  (setv vc (kwc .value-counts x :!sort :!dropna))
  (setv vc (kwc .value-counts x :!sort
    :dropna (.all (.notnull x))))
  (if (pd.core.common.is-categorical-dtype x.dtype) (do
    (setv vc (getl vc (list x.cat.categories)))
    (when (.any (.isnull x))
      ; Workaround for https://github.com/pydata/pandas/issues/9443 .
      (setv vc (.append
        (kwc pd.Series
          [(.sum (.isnull x))]
          :index ["N/A"])
        vc)))
    vc)
  ; else
    (.sort-index vc)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Matrices and DataFrames
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn cbind-join [join &rest args]
  (import [pandas :as pd])
  (setv args (list args))
  (setv index None)
  (when (and (keyword? (first args)) (= (first args) :I))
    (shift args)
    (setv index (shift args)))
  (setv chunks [])
  (while args
    (setv x (shift args))
    (if (keyword? x) (do
      (setv chunk (shift args))
      (setv chunk (if (instance? pd.Series chunk)
        (.copy chunk)
        (pd.Series chunk)))
      (setv (. chunk name) (keyword->str x))
      (.append chunks chunk))
    ; else
      (.append chunks (if (instance? pd.DataFrame x)
        x
        (pd.Series x)))))
  (setv result (kwc pd.concat :objs chunks :axis 1 :join join))
  (unless (none? index)
    (setv (. result index) index))
  result)

(defn cbind [&rest args]
  (apply cbind-join (+ (, "outer") args)))

(defn rd [a1 &optional a2]
"Round for display. Takes just a number, array, Series, or DataFrame,
or both a number of digits to round to and such an object."
  (import [numpy :as np] [pandas :as pd])
  (setv [x digits] (if (is a2 None) [a1 3] [a2 a1]))
  (cond
    [(instance? pd.DataFrame x) (do
      (setv x (.copy x))
      (for [r (range (first x.shape))]
        (for [c (range (second x.shape))]
          (when (float? (geti x r c))
            (setv (geti x r c) (round (geti x r c) digits)))))
      x)]
   [(instance? pd.Series x) (do
     (setv x (.copy x))
     (for [i (range (len x))]
       (when (float? (geti x i))
         (setv (geti x i) (round (geti x i) digits))))
     x)]
   [(instance? np.ndarray x)
     (np.round x digits)]
   [(float? x)
     (round x digits)]
   [True
     x]))

(defn with-1o-interacts [m]
"Given a data matrix m, return a matrix with a new column
for each first-order interaction. Constant columns are removed."
  (import [numpy :as np] [itertools [combinations]])
  (np.column-stack (+ [m]
    (filt (not (.all (= it (first it))))
    (lc [[v1 v2] (combinations (range (second m.shape)) 2)]
      (* (geta m : v1) (geta m : v2)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Strings
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn double-quote [s]
  (.format "\"{}\""
    (.replace (.replace s "\\" "\\\\") "\"" "\\\"")))

(defn show-expr [x]
"Stringify Hy expressions to a fairly pretty form, albeit
without newlines outside string literals."
  (import [hy [HyExpression HyDict HySymbol]])
  (cond
    [(instance? HyExpression x)
      (.format "({})" (.join " " (list (map show-expr x))))]
    [(instance? HyDict x)
      (.format "{{{}}}" (.join " " (list (map show-expr x))))]
    [(instance? HySymbol x)
      (unicode x)]
    [(instance? list x)
      (.format "[{}]" (.join " " (list (map show-expr x))))]
    [(instance? tuple x)
      (.format "(, {})" (.join " " (list (map show-expr x))))]
    [(string? x)
      (double-quote (unicode x))]
    [True
      (unicode x)]))

(defn keyword->str [x]
  (if (keyword? x)
    (slice x 2)
    x))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Lists and other basic data structures
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn concat [ll]
  (reduce (fn [accum x] (+ accum x)) ll []))

(defn seq [lo hi &optional [step 1]]
  (list (range lo (+ hi 1) step)))

(defn shift [l]
  (.pop l 0))

(defn rget [obj regex]
  (import re)
  (setv regex (re.compile regex))
  (setv keys (filt (.search regex it) (.keys obj)))
  (cond
    [(> (len keys) 1)
      (raise (LookupError "Ambiguous matcher"))]
    [(= (len keys) 0)
      (raise (LookupError "No match"))]
    [True
      (get obj (get keys 0))]))

(defn pairs [&rest a]
  (setv a (list a))
  (setv r [])
  (while a
    (.append r (, (keyword->str (shift a)) (keyword->str (shift a)))))
  r)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Higher-order functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn kfold-cv-pred [x y f &optional [n-folds 10] [shuffle True]]
"Return a np of predictions of y given x using f.

f will generally be of the form (fn [x-train y-train x-test] ...),
and should return a 1D nparray of predictions given x-test."
  (import [numpy :as np] [sklearn.cross-validation :as skcv])
  (setv y-pred (np.empty-like y))
  (for [[train-i test-i] (kwc skcv.KFold (len y) :n-folds n-folds :shuffle shuffle)]
    (setv (get y-pred test-i)
      (f (get x train-i) (get y train-i) (get x test-i))))
  y-pred)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Caching
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(setv _default-cache-dir (do
  (import os.path)
  (os.path.join (os.path.expanduser "~") ".daylight" "py-cache")))

(defn cached-eval [key f &optional bypass [cache-dir _default-cache-dir]]
"Call `f`, caching the value with the string `key`. If `bypass`
is provided, its value is written to the cache and returned
instead of calling `f` or consulting the existing cache."
  (import cPickle hashlib base64 os os.path errno time)
;  (unless (os.path.exists cache-dir)
;    (os.makedirs cache-dir))
  (setv basename (slice
    (base64.b64encode (.digest (hashlib.md5 key)) (str "+-"))
    0 -2))
  (setv path (os.path.join cache-dir basename))
  (setv value bypass)
  (when (none? value)
    (try
      (with [[o (open path "rb")]]
        (setv value (get (cPickle.load o) "value")))
      (catch [e IOError]
        (unless (= e.errno errno.ENOENT)
          (throw)))))
  (when (none? value)
    (setv value (f))
    (setv d {
      "basename" basename
      "key" key
      "value" value
      "time" (time.time)})
    (with [[o (open path "wb")]]
      (cPickle.dump d o cPickle.HIGHEST-PROTOCOL)))
  value)

(defn show-cache [&optional [cache-dir _default-cache-dir]]
"Pretty-print the caches of 'cached-eval' in chronological order."
  (import cPickle os os.path datetime)
  (setv items
    (kwc sorted :key (λ (get it "time"))
    (amap (with [[o (open (os.path.join cache-dir it) "rb")]]
      (cPickle.load o))
    (os.listdir cache-dir))))
  (for [item items]
    (print "Basename:" (get item "basename"))
    (print "Date:" (.strftime
      (datetime.datetime.fromtimestamp (get item "time"))
      "%-d %b %Y, %-I:%M:%S %p"))
    (print "Key:" (get item "key"))
    (print))
  None)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Plotting
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn dotplot [xs &optional [diam 1] ax &kwargs kwargs]
"A plot of 1D data where each value is represented as a circle,
and circles that would overlap are stacked vertically, a bit
like a histogram. Missing values are silently ignored."

  (import
    [matplotlib.pyplot :as plt]
    [matplotlib.collections [PatchCollection]]
    [numpy [isnan]])

  (setv rows [])
  (for [x (sorted (filt (and (not (isnan it)) (is-not it None)) xs))]
    (setv placed False)
    (for [row rows]
      (when (>= (- x (get row -1)) diam)
        (.append row x)
        (setv placed True)
        (break)))
    (unless placed
      (.append rows [x])))
  (setv x (flatten rows))
  (setv y (flatten (lc [[n row] (enumerate rows)] (* [(* diam (+ n .5))] (len row)))))

  (unless ax
    (setv ax (plt.gca)))
  (.set-aspect ax "equal")
  (for [side (qw left right top)]
    (.set-visible (get (. ax spines) side) False))
  
  ; Vaculously plot the points so the axes are scaled
  ; appropriately and interactive mode is respected.
  (kwc plt.scatter x y)
  ; Now add the visible circles.
  (unless (in "color" kwargs)
    (setv (get kwargs "color") "black"))
  (setv collection (apply PatchCollection
    [(lc [[x0 y0] (zip x y)] (plt.Circle (, x0 y0) (/ diam 2)))]
    kwargs))
  (.add-collection ax collection)

  (kwc .tick-params ax :!left :!labelleft)
  (kwc .set-ylim ax :bottom 0)

  collection)
