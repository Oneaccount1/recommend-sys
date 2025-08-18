package core

import (
	"archive/zip"
	"bufio"
	"fmt"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/user"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type DataSet struct {
	Ratings []float64
	Users   []int
	Items   []int
}

func NewRawSet(users, items []int, ratings []float64) DataSet {
	return DataSet{
		Users:   users,
		Items:   items,
		Ratings: ratings,
	}
}

func (d *DataSet) Length() int {
	return len(d.Ratings)
}
func (d *DataSet) Index(i int) (int, int, float64) {
	return d.Users[i], d.Items[i], d.Ratings[i]
}

func (d *DataSet) SubSet(indices []int) DataSet {
	return NewRawSet(selectInt(d.Users, indices),
		selectInt(d.Items, indices),
		selectFloat(d.Ratings, indices),
	)
}

func (d *DataSet) KFold(k int, seed int64) ([]TrainSet, []DataSet) {
	trainFolds := make([]TrainSet, k)
	testFolds := make([]DataSet, k)
	rand.New(rand.NewSource(seed))
	perm := rand.Perm(d.Length())
	foldSize := d.Length() / k
	begin, end := 0, 0
	for i := 0; i < k; i++ {
		end += foldSize
		if i < d.Length()%k {
			end++
		}
		// Test trainSet
		testIndex := perm[begin:end]
		testFolds[i] = d.SubSet(testIndex)
		// Train trainSet
		trainIndex := concatenate(perm[0:begin], perm[end:d.Length()])
		trainFolds[i] = NewTrainSet(d.SubSet(trainIndex))
		begin = end
	}
	return trainFolds, testFolds
}

func (d *DataSet) Split(testSize float64, seed int64) (TrainSet, DataSet) {
	rand.New(rand.NewSource(0))
	perm := rand.Perm(d.Length())
	mid := int(float64(d.Length()) * testSize)
	testSet := d.SubSet(perm[:mid])
	trainSet := d.SubSet(perm[mid:])
	return NewTrainSet(trainSet), testSet
}

// ToCSV Save data set to csv.
func (d *DataSet) ToCSV(fileName string, sep string) error {
	file, err := os.Create(fileName)
	defer file.Close()
	if err == nil {
		writer := bufio.NewWriter(file)
		for i := range d.Ratings {
			writer.WriteString(fmt.Sprintf("%v%s%v%s%v\n",
				d.Users[i], sep,
				d.Items[i], sep,
				d.Ratings[i]))
		}
	}
	return err
}

// Predict ratings for a set of <userId, itemId>s.
func (d *DataSet) Predict(estimator Estimator) []float64 {
	predictions := make([]float64, d.Length())
	for j := 0; j < d.Length(); j++ {
		userId, itemId, _ := d.Index(j)
		predictions[j] = estimator.Predict(userId, itemId)
	}
	return predictions
}

type Set map[int]interface{}

type TrainSet struct {
	DataSet
	GlobalMean float64
	UserCount  int
	ItemCount  int

	InnerUserIDs map[int]int
	InnerItemIDs map[int]int
	outerUserIDs []int
	outerItemIDs []int

	userRatings [][]IDRating
	itemRatings [][]IDRating
}

type IDRating struct {
	ID     int
	Rating float64
}

const newID = -1

func NewTrainSet(rowSet DataSet) TrainSet {
	set := TrainSet{}
	set.DataSet = rowSet
	set.GlobalMean = stat.Mean(rowSet.Ratings, nil)

	// 创建userID -> innerUserID的映射
	set.InnerUserIDs = make(map[int]int)
	for _, userID := range set.Users {
		if _, exist := set.InnerUserIDs[userID]; !exist {
			set.InnerUserIDs[userID] = set.UserCount
			set.UserCount++
		}
	}
	// 创建itemID -> innerItemID的映射
	set.InnerItemIDs = make(map[int]int)
	for _, itemID := range set.Items {
		if _, exist := set.InnerItemIDs[itemID]; !exist {
			set.InnerItemIDs[itemID] = set.ItemCount
			set.ItemCount++
		}
	}

	return set
}

// UserSet   获取数据集所有User
func (set *TrainSet) UserSet() Set {
	return unique(set.Users)
}

// ItemSet   获取所有数据集Item
func (set *TrainSet) ItemSet() Set {
	return unique(set.Items)
}

// RatingRange 获取数据集评分的范围
func (set *TrainSet) RatingRange() (float64, float64) {
	return floats.Min(set.Ratings), floats.Max(set.Ratings)
}

func (set *TrainSet) ConvertUserID(userID int) int {
	if innerUserID, exist := set.InnerUserIDs[userID]; exist {
		return innerUserID
	}
	return newID
}
func (set *TrainSet) ConvertItemID(itemID int) int {
	if innerItemID, exist := set.InnerItemIDs[itemID]; exist {
		return innerItemID
	}
	return newID
}

// UserRatings Get users' LeftRatings: an array of <itemId, rating> for each user.
func (set *TrainSet) UserRatings() [][]IDRating {
	if set.userRatings == nil {
		set.userRatings = make([][]IDRating, set.UserCount)

		for innerUserID := range set.userRatings {
			set.userRatings[innerUserID] = make([]IDRating, 0)
		}
		for i := 0; i < len(set.Users); i++ {
			innerUserID := set.ConvertUserID(set.Users[i])
			innerItemID := set.ConvertItemID(set.Items[i])
			set.userRatings[innerUserID] = append(set.userRatings[innerUserID], IDRating{ID: innerItemID, Rating: set.Ratings[i]})
		}
	}
	return set.userRatings
}

// ItemRatings Get items' LeftRatings: an array of <userId, Rating> for each item.
func (set *TrainSet) ItemRatings() [][]IDRating {
	if set.itemRatings == nil {
		set.itemRatings = make([][]IDRating, set.ItemCount)
		for i := range set.itemRatings {
			set.itemRatings[i] = make([]IDRating, 0)
		}

		for i := 0; i < len(set.Items); i++ {
			innerUserId := set.ConvertUserID(set.Users[i])
			innerItemId := set.ConvertItemID(set.Items[i])
			set.itemRatings[innerItemId] = append(set.itemRatings[innerItemId], IDRating{ID: innerUserId, Rating: set.Ratings[i]})
		}
	}
	return set.itemRatings
}

func (set *TrainSet) Split(testSize int) (TrainSet, TrainSet) {
	return TrainSet{}, TrainSet{}
}

func means(a [][]IDRating) []float64 {
	ret := make([]float64, len(a))
	// 注意Nan
	for i := range a {
		sum, count := 0.0, 0.0

		for _, ir := range a[i] {
			sum += ir.Rating
			count++
		}
		ret[i] = sum / count
	}
	return ret
}
func sorts(rating [][]IDRating) []SortedIdRatings {
	a := make([]SortedIdRatings, len(rating))
	for i := range rating {
		a[i] = SortedIdRatings{rating[i]}
		sort.Sort(a[i])
	}
	return a
}

type SortedIdRatings struct {
	data []IDRating
}

func NewSortedIdRatings(a []IDRating) SortedIdRatings {
	b := SortedIdRatings{a}
	sort.Sort(b)
	return b
}

func (sir SortedIdRatings) Len() int {
	return len(sir.data)
}

func (sir SortedIdRatings) Swap(i, j int) {
	sir.data[i], sir.data[j] = sir.data[j], sir.data[i]
}

func (sir SortedIdRatings) Less(i, j int) bool {
	return sir.data[i].ID < sir.data[j].ID
}

/* loader */

// Load build in data set
func LoadDataFromBuiltIn(dataSetName string) DataSet {
	// Extract data set information
	dataSet, exist := builtInDataSets[dataSetName]
	if !exist {
		log.Fatal("no such data set", dataSetName)
	}
	const dataFolder = "data"
	const tempFolder = "temp"
	dataFileName := filepath.Join(dataFolder, dataSet.path)
	if _, err := os.Stat(dataFileName); os.IsNotExist(err) {
		zipFileName, _ := downloadFromUrl(dataSet.url, tempFolder)
		unzip(zipFileName, dataFolder)
	}
	return LoadDataFromFile(dataFileName, dataSet.sep)
}

// Load data from file
func LoadDataFromFile(fileName string, sep string) DataSet {
	users := make([]int, 0)
	items := make([]int, 0)
	ratings := make([]float64, 0)
	// Open file
	file, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	// Read CSV file
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Split(line, sep)
		user, _ := strconv.Atoi(fields[0])
		item, _ := strconv.Atoi(fields[1])
		rating, _ := strconv.Atoi(fields[2])
		users = append(users, user)
		items = append(items, item)
		ratings = append(ratings, float64(rating))
	}
	return NewRawSet(users, items, ratings)
}

// Download file from URL.
func downloadFromUrl(src string, dst string) (string, error) {
	// Extract file name
	tokens := strings.Split(src, "/")
	fileName := filepath.Join(dst, tokens[len(tokens)-1])
	// Create file
	if err := os.MkdirAll(filepath.Dir(fileName), os.ModePerm); err != nil {
		return fileName, err
	}
	output, err := os.Create(fileName)
	if err != nil {
		fmt.Println("Error while creating", fileName, "-", err)
		return fileName, err
	}
	defer output.Close()
	// Download file
	response, err := http.Get(src)
	if err != nil {
		fmt.Println("Error while downloading", src, "-", err)
		return fileName, err
	}
	defer response.Body.Close()
	// Save file
	_, err = io.Copy(output, response.Body)
	if err != nil {
		fmt.Println("Error while downloading", src, "-", err)
		return fileName, err
	}
	return fileName, nil
}

// Unzip zip file.
func unzip(src string, dst string) ([]string, error) {
	var fileNames []string
	// Open zip file
	r, err := zip.OpenReader(src)
	if err != nil {
		return fileNames, err
	}
	defer r.Close()
	// Extract files
	for _, f := range r.File {
		// Open file
		rc, err := f.Open()
		if err != nil {
			return fileNames, err
		}
		// Store filename/path for returning and using later on
		filePath := filepath.Join(dst, f.Name)
		// Check for ZipSlip. More Info: http://bit.ly/2MsjAWE
		if !strings.HasPrefix(filePath, filepath.Clean(dst)+string(os.PathSeparator)) {
			return fileNames, fmt.Errorf("%s: illegal file path", filePath)
		}
		// Add filename
		fileNames = append(fileNames, filePath)

		if f.FileInfo().IsDir() {
			// Create folder
			os.MkdirAll(filePath, os.ModePerm)
		} else {
			// Create all folders
			if err = os.MkdirAll(filepath.Dir(filePath), os.ModePerm); err != nil {
				return fileNames, err
			}
			// Create file
			outFile, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
			if err != nil {
				return fileNames, err
			}
			// Save file
			_, err = io.Copy(outFile, rc)
			// Close the file without defer to close before next iteration of loop
			outFile.Close()
			if err != nil {
				return fileNames, err
			}
		}
		// Close file
		rc.Close()
	}
	return fileNames, nil
}

/* Built-in */

// Built-in data set
type _BuiltInDataSet struct {
	url  string
	path string
	sep  string
}

var builtInDataSets = map[string]_BuiltInDataSet{
	"ml-100k": {
		url:  "https://cdn.sine-x.com/datasets/movielens/ml-100k.zip",
		path: "ml-100k/u.data",
		sep:  "\t",
	},
	"ml-1m": {
		url:  "https://cdn.sine-x.com/datasets/movielens/ml-1m.zip",
		path: "ml-1m/ratings.dat",
		sep:  "::",
	},
	"ml-10m": {
		url:  "https://cdn.sine-x.com/datasets/movielens/ml-10m.zip",
		path: "ml-10M100K/ratings.dat",
		sep:  "::",
	},
	"ml-20m": {
		url:  "https://cdn.sine-x.com/datasets/movielens/ml-20m.zip",
		path: "ml-20m/ratings.csv",
		sep:  ",",
	},
}

// The data directories
var (
	downloadDir string
	dataSetDir  string
	tempDir     string
)

func init() {
	usr, _ := user.Current()
	gorseDir := usr.HomeDir + "/.gorse"
	downloadDir = gorseDir + "/download"
	dataSetDir = gorseDir + "/datasets"
	tempDir = gorseDir + "/temp"
}
