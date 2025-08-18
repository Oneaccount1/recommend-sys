package core

import (
	"bytes"
	"encoding/gob"
	"os"
	"path/filepath"
)

// Load a object from file.
func Load(fileName string, object interface{}) error {
	file, err := os.Open(fileName)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	return decoder.Decode(object)
}

// Save a object to file.
func Save(fileName string, object interface{}) error {
	// 创建目录
	if err := os.MkdirAll(filepath.Dir(fileName), os.ModePerm); err != nil {
		return err
	}
	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(object)
}
func Copy(dst, src interface{}) {
	buffer := new(bytes.Buffer)
	encoder := gob.NewEncoder(buffer)
	encoder.Encode(src)
	decoder := gob.NewDecoder(buffer)
	decoder.Decode(dst)
}
