var nodemailer = require('nodemailer');

var transporter = nodemailer.createTransport({
service: 'Gmail',
  auth: {
      user: 'studdb.db@gmail.com',
      pass: 'student_db'
  }
});


var mailOptions = {
  to: 'vigneshj6@gmail.com', // list of receivers 
  subject: 'Password Reset', // Subject line 
  html: 'Your one time password is', // html body 
  attachments: [
    {
      path:'./FaceRecReview2.pdf'
    }
    ]
};

transporter.sendMail(mailOptions, function (error, info) {
console.log(error,info);
});