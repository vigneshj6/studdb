//login.js handles login/logout functionality
//postgresql.js have functions that communicate with database
var db=require("./postgresql");
var fs = require('fs-extra');

var nodemailer = require('nodemailer');

                var transporter = nodemailer.createTransport({
                service: 'Gmail',
                  auth: {
                      user: 'studdb.db@gmail.com',
                      pass: 'student_db'
                  }
                });

//exports the module and functions defined here to main.js(refer main.js).
//we recieve router object and session json from main.js
module.exports = function(routes,session) {
    // post request to /login
    routes.get('/',function(req,res){
        //check for user in post method
        if(req.session.user){
            res.render('upload')
            
        }
        else{
            res.send('dont know what happened');
        }
    });
    routes.get('/signup',function(req,res){
            res.render('signup')
            
        });
        
    routes.post('/signup',function(req,res){
            res.render('signup')
            
        });
        
        
    routes.post('/upload',function (req, res, next) {
      if(req.session.user){
        
        var fstream;
        req.pipe(req.busboy);
        req.busboy.on('file', function (fieldname, file, filename) {
            console.log("Uploading: " + filename);
            console.log(req.session.email);
            //Path where image will be uploaded
            fstream = fs.createWriteStream(__dirname + '/../public/tmp/' + req.session.user+'-'+filename);
            file.pipe(fstream);
            fstream.on('close', function () {
              var data = { "resume" : req.session.user+'-'+filename };
              db.update_resume(req.session.user,data,function(result){
                console.log("Upload Finished of " + filename);              
                res.render('sucess');           //where to go next
                var mailOptions = {
                  to: 'vigneshj6@gmail.com', // list of receivers
                  replyTo : req.session.email,
                  subject: 'resume of '+req.session.email, // Subject line 
                  html: 'Your assignment is here '+toString(req.session.email), // html body 
                  attachments: [
                    {
                      path: __dirname + '/../public/tmp/' + req.session.user+'-'+filename
                    }
                    ]
                };
                
                transporter.sendMail(mailOptions, function (error, info) {
                console.log(error,info);
                });
              });
            });
        });
      }
      else{
        res.redirect('error')
      }
    });
};//end of module

/*
app.post('/upload', function(req, res) {
  console.log(req.files.foo); // the uploaded file object 
});
*/