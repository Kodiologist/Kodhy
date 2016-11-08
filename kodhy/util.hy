(require [kodhy.macros [*]])

(setv T True)
(setv F False)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Numbers and arrays
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn signum [x] (cond
  [(< x 0) -1]
  [(> x 0)  1]
  [T        0]))

(defn product [l]
  (setv a 1)
  (for [x l]
    (*= a x))
  a)

(defn odds-ratio [p1 p2]
  (/ (* p1 (- 1 p2)) (* p2 (- 1 p1))))

(defn logit [x]
  (import numpy)
  (numpy.log (/ x (- 1 x))))

(defn ilogit [x]
  (import numpy)
  (/ 1 (+ 1 (numpy.exp (- x)))))

(defn zscore [x]
  (/ (- x (.mean x)) (.std x :ddof 0)))

(defn hzscore [x]
"Half the z-score. Divides by two SDs instead of one, per:
Gelman, A. (2008). Scaling regression inputs by dividing by two standard deviations. Statistics in Medicine, 27(15), 2865–2873. doi:10.1002/sim.3107"
  (/ (- x (.mean x)) (* 2 (.std x :ddof 0))))

(defn rmse [v1 v2]
"Root mean square error."
  (import [numpy :as np])
  (np.sqrt (np.mean (** (- v1 v2) 2))))

(defn mean-ad [v1 v2]
"Mean absolute deviation."
  (import [numpy :as np])
  (np.mean (np.abs (- v1 v2))))

(defn valcounts [x &optional y]
  (import [pandas :as pd] [numpy :as np])
  (setv [x y] (rmap [v [x y]]
    (if (or (none? v) (instance? pd.Series v))
      v
      (pd.Series (list v)))))
  (if (none? y)
    (.rename (.value-counts x :sort F :dropna F)
      (λ (if (pd.isnull it) (str "N/A") it)))
    (pd.crosstab
      (.replace x np.nan "~N/A") (.replace y np.nan "~N/A"))))

(defn weighted-choice [l]
; The argument should be a list of (weight, object) pairs.
; http://stackoverflow.com/a/3679747
  (import random)
  (setv r (random.uniform 0 (sum (map first l))))
  (for [[w x] l]
    (-= r w)
    (when (<= r 0)
      (break)))
  x)

(defn pds-from-pairs [l &kwargs kwargs]
  (import [pandas :as pd])
  (apply pd.Series [(amap (second it) l) (amap (first it) l)] kwargs))

(defn pd-posix-time [series]
  (import [numpy :as np])
  (// (.astype series np.int64) (int 1e9)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Matrices and DataFrames
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn cbind-join [join &rest args]
  (import [pandas :as pd])
  (setv args (list args))
  (setv index None)
  (when (and (string? (first args)) (= (first args) "I"))
    (shift args)
    (setv index (shift args)))
  (defn scalar? [x]
    (or (string? x) (not (iterable? x))))
  (setv height (max (amap
    (if (scalar? it) 1 (len it))
    args)))
  (setv chunks [])
  (while args
    (setv x (shift args))
    (if (string? x)
      (do
        (setv chunk (shift args))
        (setv chunk (cond
          [(instance? pd.Series chunk)
            (.copy chunk)]
          [(scalar? chunk)
            (pd.Series (* [chunk] height))]
          [T
            (pd.Series chunk)]))
        (setv chunk.name x)
        (.append chunks chunk))
      (.append chunks (cond
          [(instance? pd.DataFrame x)
            x]
          [(scalar? x)
            (pd.Series (* [x] height))]
          [T
            (pd.Series x)]))))
  (setv result (pd.concat :objs chunks :axis 1 :join join))
  (unless (none? index)
    (setv (. result index) index))
  result)

(defn cbind [&rest args]
  (apply cbind-join (+ (, "outer") args)))

(defn df-from-pairs [l]
  (import [pandas :as pd])
  (setv d (pd.DataFrame (lc [row l] (lc [[_ v] row] v))))
  (setv d.columns (amap (first it) (first l)))
  d)

(defn drop-unused-cats [d &optional [inplace F]]
  ; Drops unused categories from all categorical columns.
  ; Can also be applied to a Series.
  (import [pandas :as pd])
  (unless inplace
    (setv d (.copy d)))
  (for [[_ col] (if (instance? pd.Series d) [[None d]] (.iteritems d))]
    (when (hasattr col "cat")
      (.remove-unused-categories col.cat :inplace T)))
  d)

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
   [T
     x]))

(defn with-1o-interacts [m &optional column-names]
"Given a data matrix m, return a matrix with a new column
for each first-order interaction. Constant columns are removed."
  (import [numpy :as np] [itertools [combinations]])
  (when column-names
    (assert (= (len column-names) (second m.shape))))
  (setv [new-names new-cols] (apply zip (filt
    (not (.all (= (second it) (first (second it)))))
    (lc [[v1 v2] (combinations (range (second m.shape)) 2)] (,
      (when column-names
        (tuple (sorted [(get column-names v1) (get column-names v2)])))
      (np.multiply (geta m : v1) (geta m : v2)))))))
  (setv new-m (np.column-stack (+ (, m) new-cols)))
  (if column-names
    (, new-m (+ (tuple column-names) new-names))
    new-m))

(defn print-big-pd [obj]
  (import [pandas :as pd])
  (with [(pd.option-context
      "display.max_rows" (int 5000)
      "display.max_columns" (int 100)
      "display.width" (int 1000)
      "display.max_colwidth" (int 500))]
    (print obj)))

(defn pd-to-pretty-json [path df]
  ; Serializes a Pandas dataframe to an obvious-looking JSON format.
  ; Information about categorial columns is saved as metadata.
  (import [numpy :as np] [collections [OrderedDict]])
  (setv out {})

  (setv (get out "categories") (OrderedDict (rmap [col (ssi df.dtypes (= $ "category"))]
    [col {
      "ordered" (. (getl df : col) cat ordered)
      "categories" (list (. (getl df : col) cat categories))}])))

  (setv cols (list df.columns))
  (setv (get out "table") (.astype df.values object))
  (setv (get out "first_col_is_row_labels") F)
  (when (or df.index.name
      (not (.all (= df.index (list (range (len df)))))))
    ; We only include the index as a column if it has a name or
    ; is something other than consecutive integers starting from
    ; 0.
    (setv (get out "first_col_is_row_labels") T)
    (setv (get out "table") (np.column-stack [df.index (get out "table")]))
    (setv cols (+ [df.index.name] cols)))
  (setv (get out "table") (+ [cols] (.tolist (get out "table"))))

  (if (none? path)
    (json-dumps-pretty out)
    (barf path (json-dumps-pretty out))))

(defn pretty-json-to-pd [path]
  (import json [pandas :as pd])
  (setv j (json.loads (slurp path)))
  (setv df (pd.DataFrame (cut (get j "table") 1)
    :columns (get j "table" 0)))
  (when (get j "first_col_is_row_labels")
    (setv df (.set-index df (first df.columns))))
  (for [[catcol meta] (.items (.get j "categories" {}))]
    (setv (getl df : catcol)
      (apply .astype [(getl df : catcol) "category"] meta)))
   df)

;(defn pd-to-pretty-json [path df]
;  ; Serializes a Pandas dataframe to an obvious-looking JSON format.
;  ; Categorical columns are converted to codes, but the original
;  ; categories are provided as metadata.
;  (import [numpy :as np])
;  (setv df (.copy df))
;  (setv out {})
;
;  (setv catcols (ssi df.dtypes (= $ "category")))
;  (setv (get out "categories") (dict (rmap [col catcols]
;    [col [(if (. (getl df : col) cat ordered) "ordered" "unordered")
;      (list (. (getl df : col) cat categories))]])))
;  (for [col catcols]
;    (setv (getl df : col) (. (getl df : col) cat codes)))
;
;  (setv cols (list df.columns))
;  (setv (get out "table") (.astype df.values object))
;  (when (or df.index.name
;      (not (.all (= df.index (list (range (len df)))))))
;    ; We only include the index as a column if it has a name or
;    ; is something other than consecutive integers starting from
;    ; 0.
;    (setv (get out "table") (np.column-stack [df.index (get out "table")]))
;    (setv cols (+ [(or df.index.name "I")] cols)))
;  (setv (get out "table") (+ [cols] (.tolist (get out "table"))))
;
;  (if path
;    (barf path (json-dumps-pretty out))
;    (json-dumps-pretty out)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Strings
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn cat [&rest args &kwargs kwargs]
  (.join
    (or (.get kwargs "sep") "")
    (lc [x args] x (string x))))

(defn ucfirst [s]
  (and s (+ (.upper (first s)) (cut s 1))))

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
    [(keyword? x)
      (+ ":" (keyword->str x))]
    [(instance? HySymbol x)
      (unicode x)]
    [(instance? list x)
      (.format "[{}]" (.join " " (list (map show-expr x))))]
    [(instance? tuple x)
      (.format "(, {})" (.join " " (list (map show-expr x))))]
    [(string? x)
      (double-quote (unicode x))]
    [T
      (unicode x)]))

(defn keyword->str [x]
  (if (keyword? x)
    (cut x 2)
    x))

(defn str->keyword [x]
  (if (keyword? x)
    x
    (+ "\ufdd0:" x)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Lists and other basic data structures
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn concat [ll]
  (reduce (fn [accum x] (+ accum x)) ll []))

(defn merge-dicts [&rest ds]
  (setv out {})
  (for [d ds]
    (.update out d))
  out)

(defn seq [lo hi &optional [step 1]]
  (list (range lo (+ hi step) step)))

(defn shift [l]
  (.pop l 0))

(defn unique [l]
  (setv seen (set))
  (filt (when (not-in it seen) (.add seen it) T) l))

(defn all-unique? [l]
  (= (len l) (len (set l))))

(defn mins [iterable &optional [key (λ it)] comparator-fn [agg-fn min]]
  ; Returns a list of minimizing values of the iterable,
  ; in their original order.
  (unless comparator-fn
    (import operator)
    (setv comparator-fn operator.le))
  (setv items (list iterable))
  (if items
    (do
      (setv vals (list (map key items)))
      (setv vm (agg-fn vals))
      (lc [[item val] (zip items vals)]
        (comparator-fn val vm)
        item))
    []))

(defn maxes [iterable &optional [key (λ it)]]
  (import operator)
  (mins iterable key operator.ge max))

(defn rget [obj regex]
  (import re)
  (setv regex (re.compile regex))
  (setv keys (filt (.search regex it) (.keys obj)))
  (cond
    [(> (len keys) 1)
      (raise (LookupError "Ambiguous matcher"))]
    [(= (len keys) 0)
      (raise (LookupError "No match"))]
    [T
      (get obj (get keys 0))]))

(defn pairs [&rest a]
  (setv a (list a))
  (setv r [])
  (while a
    (.append r (, (keyword->str (shift a)) (keyword->str (shift a)))))
  r)

(defn by-ns [n iterable]
  (apply zip (* [(iter iterable)] n)))
(defn by-2s [iterable]
  (by-ns 2 iterable))

(defn iter-with-prev (iterable)
  (setv prev None)
  (for [item iterable]
    (yield (, prev item))
    (setv prev item)))

(defn iter-with-prev1 (iterable)
  ; Like iter-with-prev, but skips the first pair, which has
  ; None for the previous value.
  (import itertools)
  (itertools.islice (iter-with-prev iterable) 1 None))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Files
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn slurp [name &optional mode encoding buffering]
  (setv f open)
  (when encoding
    (import codecs)
    (setv f codecs.open))
  (with [o (apply f [name] (dict (+
      (if (none? mode)      [] [(, "mode" mode)])
      (if (none? encoding)  [] [(, "encoding" encoding)])
      (if (none? buffering) [] [(, "buffering" buffering)]))))]
    (o.read)))

(defn barf [name content &optional [mode "w"] encoding buffering]
  (setv f open)
  (when encoding
    (import codecs)
    (setv f codecs.open))
  (with [o (apply f [name mode] (dict (+
      (if (none? encoding)  [] [(, "encoding" encoding)])
      (if (none? buffering) [] [(, "buffering" buffering)]))))]
    (o.write content)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * JSON
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn json-dumps-pretty [o &kwargs kwargs]
  ; Like json.dumps, but arrays or objects of atomic values are
  ; printed without internal indents, and with different
  ; option defaults.
  (import json uuid)
  (for [[option value] (pairs
      :indent 2 :separators (, "," ": ") :sort_keys T)]
    (when (none? (.get kwargs option))
      (setv (get kwargs option) value)))
  (setv substituted-parts {})
  (defn recursive-subst [x]
    ; Replaces lists or dictionaries of atomic values with UUID
    ; strings.
    (if (isinstance x (, list tuple dict))
      (if (all (lc [v (if (isinstance x dict) (.values x) x)]
            (isinstance v (, bool (type None) int long float str unicode))))
        (do
          (setv my-id (. (uuid.uuid4) hex))
          (setv (get substituted-parts my-id) x)
          my-id)
        (if (isinstance x dict)
          (dict (lc [[k v] (.items x)] (, k (recursive-subst v))))
          (lc [v x] (recursive-subst v))))
      x))
  (setv json-str (apply json.dumps [(recursive-subst o)] kwargs))
  (setv (get kwargs "indent") None)
  (setv (get kwargs "separators") (, ", " ": "))
  (for [[my-id x] (.items substituted-parts)]
    (setv json-str (.replace json-str (+ "\"" my-id "\"")
      (.rstrip (apply json.dumps [x] kwargs))
      1)))
  json-str)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Cross-validation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn kfold-cv-pred [x y f &optional [n-folds 10] [shuffle T] [random-state None] folds]
"Return a np of predictions of y given x using f. x is expected to be
a numpy matrix, not a pandas DataFrame.

f will generally be of the form (fn [x-train y-train x-test] ...),
and should return a 1D nparray of predictions given x-test."
  (import [numpy :as np])
  (setv y-pred None)
  (unless folds
    (import [sklearn.model-selection :as skms])
    (setv folds (.split
      (skms.KFold :n-splits n-folds
        :shuffle shuffle :random-state random-state)
      x)))
  (for [[train-i test-i] folds]
    (setv result (f (get x train-i) (get y train-i) (get x test-i)))
    (when (none? y-pred)
      (setv y-pred (np.empty-like y :dtype result.dtype)))
    (setv (get y-pred test-i) result))
  y-pred)

(defn choose-labeled-cv-folds [subjects labels bin-label-possibilities]
; Put the subjects into cross-validation folds such that all the
; subjects with a given label are in the same fold.
; bin-label-possibilities should be the return value of
; bin-labels.
  (import [random [choice shuffle]] [collections [Counter]])
  (setv group-sizes (Counter labels))
  (setv bins (list (choice bin-label-possibilities)))
  (shuffle bins)
  (amap
    (concat (amap
      (do
        (setv target-size it)
        (setv label (first (afind (= (second it) target-size) (.items group-sizes))))
        (del (get group-sizes label))
        (lc [[s l] (zip subjects labels)] (= l label) s))
      it))
    bins))

(defn bin-labels [labels &optional [n-bins 10] max-bin-size]
; A routine to prepare input for choose-labeled-cv-folds.
;
; Finds ways to sort a list of label objects (which is
; expected to have lots of duplicates) into a fixed number
; bins in a way that gets the bin sizes as close to equal
; as possible. This is just the multiprocessor-scheduling problem
; ( https://en.wikipedia.org/wiki/Multiprocessor_scheduling )
; with a different metric to optimize.
;
; The algorithm simply enumerates all possibilities, so it will
; will be too slow with large inputs.
;
; Returns a tuple of possibilities, each of which is a tuple of
; bins. Each bin is a tuple of numbers representing the size
; of a labelled group.

  (import [collections [Counter]])

  (setv initial-state (,
    (tuple (sorted (.values (Counter labels))))
    (* (, (,)) n-bins)))

  (setv states (set [initial-state]))
  (setv explore [initial-state])
  (setv iteration 0)
  (setv explore-len-was (len explore))

  (print "Generating possible arrangements")
  (while explore
    (+= iteration 1)
    (unless (% iteration 10000)
      (print (len explore) (.format "({:+d})" (- (len explore) explore-len-was)))
      (setv explore-len-was (len explore)))
    (setv [remaining bins] (.pop explore 0))
    (setv rseen (set))
    (for [r-i (range (len remaining))]
      (setv x (get remaining r-i))
      (when (in x rseen)
        (continue))
      (.add rseen x)
      (setv new-remaining (+ (cut remaining 0 r-i) (cut remaining (+ r-i 1))))
      (for [b-i (range n-bins)]
        (setv new-bin (+ (get bins b-i) (, x)))
        (when (and max-bin-size (> (sum new-bin) max-bin-size))
          (continue))
        (setv new-bins (tuple (sorted (+
          (cut bins 0 b-i)
          (, (tuple (sorted new-bin)))
          (cut bins (+ b-i 1))))))
        (setv new-state (, new-remaining new-bins))
        (when (in new-state states)
          (continue))
        (.add states new-state)
        (.append explore new-state))))

  (print "Finding minima")
  (setv target-bin-size (/ (len labels) n-bins))
  (mins
    (lc [[remaining bins] states] (and (not remaining) (all bins)) bins)
    (λ (sum (amap (** (- (sum it) target-bin-size) 2) it)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Caching
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(setv _default-cache-dir (do
  (import os.path)
  (os.path.join (os.path.expanduser "~") ".daylight" "py-cache")))

(defn cached-eval [key f &optional bypass cache-dir]
"Call `f`, caching the value with the string `key`. If `bypass`
is provided, its value is written to the cache and returned
instead of calling `f` or consulting the existing cache."
  (import cPickle hashlib base64 os os.path errno time)
;  (unless (os.path.exists cache-dir)
;    (os.makedirs cache-dir))
  (unless cache-dir
    (setv cache-dir _default-cache-dir))
  (setv basename (cut
    (base64.b64encode (.digest (hashlib.md5 key)) (str "+_"))
    0 -2))
  (setv path (os.path.join cache-dir basename))
  (setv value bypass)
  (setv write-value T)
  (when (none? value)
    (try
      (do
        (with [o (open path "rb")]
          (setv value (get (cPickle.load o) "value")))
        (setv write-value F))
      (except [e IOError]
        (unless (= e.errno errno.ENOENT)
          (throw)))))
  (when (none? value)
    (setv value (f)))
  (when write-value
    (setv d {
      "basename" basename
      "key" key
      "value" value
      "time" (time.time)})
    (with [o (open path "wb")]
      (cPickle.dump d o cPickle.HIGHEST-PROTOCOL)))
  value)

(defn show-cache [&optional [cache-dir _default-cache-dir]]
"Pretty-print the caches of 'cached-eval' in chronological order."
  (import cPickle os os.path datetime)
  (setv items
    (sorted :key (λ (get it "time"))
    (amap (with [o (open (os.path.join cache-dir it) "rb")]
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
;; * Support for Tversky
;; https://github.com/Kodiologist/Tversky
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn tversky-format-s [s]
  (amap (.format "s{:03d}" it) s))

(defn unpack-tversky [db-path &optional
    [include-incomplete T]
    exclude-sns]
  (import sqlite3 [pandas :as pd])
  (try
    (do
      (setv db (sqlite3.connect db-path))
      (.execute db "pragma foreign_keys = on")

      (setv sb (pd.read-sql-query :con db :index-col "sn"
        :parse-dates (dict (amap (, it {"unit" "s"}) (qw consented_t began_t completed_t)))
        "select
            sn, experimenter, ip, task,
            consented_t,
            began_t,
            case when completed_t = 'assumed' then null else completed_t end
              as completed_t,
            MTurk.hitid as hit,
            MTurk.assignmentid as asgmt,
            MTurk.workerid as worker
        from Subjects
            left join
                (select sn, min(first_sent) as began_t from Timing group by sn)
                using (sn)
            left join MTurk using (sn)"))
      ; Make some columns categorical, with the levels ordered
      ; chronologically.
      (for [c (qw experimenter ip hit task)]
        (setv (getl sb : c) (pd.Categorical
          (getl sb : c)
          :ordered T
          :categories (list (.unique (.dropna (getl (.sort-values sb "began_t") : c)))))))
      (setv ($ sb tv) (+ 1 (. ($ sb task) cat codes)))
        ; "tv" for "task version".
      (when exclude-sns
        (setv sb (.drop sb exclude-sns)))
      (unless include-incomplete
        (setv sb (.dropna sb :subset ["completed_t"])))

      (.execute db "create temporary table IncludeSN(sn integer primary key)")
      (.executemany db "insert into IncludeSN (sn) values (?)" (amap (, it) sb.index))

      (setv dat (.sortlevel (geti (pd.read-sql-query :con db :index-col ["sn" "k"]
        "select * from D where sn in (select * from IncludeSN)") : 0)))

      (setv timing (.sortlevel (pd.read-sql-query :con db :index-col ["sn" "k"]
        :parse-dates (dict (amap (, it {"unit" "s"}) (qw first_sent received)))
        "select * from Timing where sn in (select * from IncludeSN)")))

      (setv sb.index (tversky-format-s sb.index))
      (for [df [dat timing]]
        (.set-levels df.index :inplace T :level "sn"
          (tversky-format-s (first df.index.levels))))

      (, sb dat timing))

    (finally
      (.close db))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; * Support for kodhy.macros
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defcls _KodhyBlockReturn [Exception]
  __init__ (meth [block-name value]
    (setv @block-name block-name)
    (setv @value value)))

(defn ret [&optional value]
"Return from the innermost 'block'."
  (raise (_KodhyBlockReturn None value)))

(defn ret-from [block-name &optional value]
"Return from the innermost 'block' with the given name."
  (raise (_KodhyBlockReturn block-name value)))

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
    (setv placed F)
    (for [row rows]
      (when (>= (- x (get row -1)) diam)
        (.append row x)
        (setv placed T)
        (break)))
    (unless placed
      (.append rows [x])))
  (setv x (flatten rows))
  (setv y (flatten (lc [[n row] (enumerate rows)] (* [(* diam (+ n .5))] (len row)))))

  (unless ax
    (setv ax (plt.gca)))
  (.set-aspect ax "equal")
  (for [side (qw left right top)]
    (.set-visible (get (. ax spines) side) F))
  
  ; Vaculously plot the points so the axes are scaled
  ; appropriately and interactive mode is respected.
  (plt.scatter x y)
  ; Now add the visible markers.
  (unless (in "color" kwargs)
    (setv (get kwargs "color") "black"))
  (setv collection (apply PatchCollection
    (if (.pop kwargs "rect" F)
      [(lc [[x0 y0] (zip x y)] (plt.Rectangle (, (- x0 (/ diam 2)) (- y0 (/ diam 2))) diam diam))]
      [(lc [[x0 y0] (zip x y)] (plt.Circle (, x0 y0) (/ diam 2)))])
    kwargs))
  (.add-collection ax collection)

  (.tick-params ax :left F :labelleft F)
  (.set-ylim ax :bottom 0)

  collection)

(defn rectplot [xs &optional [diam 1] ax &kwargs kwargs]
"`dotplot` using rectangles instead of circles."
  (setv (get kwargs "rect") T)
  (apply dotplot [xs diam ax] kwargs))
