const corpusFolder = '/home/taxonfinder/corpus';
const fs = require('fs');
const glob = require('glob')
const taxonfinder = require('taxonfinder');

glob(corpusFolder + '/*.txt', {}, (err, files)=>{
  files.forEach(file => {
    fs.readFile(file, 'utf-8', (err, data) => { 
       if (err) throw err; 
  
       // Converting Raw Buffer to text 
       // data using tostring function. 
       var resultsWithOffsets = taxonfinder.findNamesAndOffsets(data);

       console.log(file);
       console.log(resultsWithOffsets);
    }) 
  })
})
