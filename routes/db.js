var fs = require("fs");

var pg = require("pg");

//Login
var login = fs.readFileSync('./query/login.sql').toString();

//Update Resume
var update_resume= fs.readFileSync('./query/update_resume.sql').toString();

var config = {
    postgres:{
        user:'ubuntu',
        password:'enter',
        host:'localhost'
    }
};

function db(db){
    var conString;
    if(db!='')
    {
        conString = process.env.DATABASE_URL||("pg://"+config.postgres.user+":"+config.postgres.password+"@"+config.postgres.host+":5432/");
        conString = conString+db;
    }
    else
    {
        conString = "pg://"+config.postgres.user+":"+config.postgres.password+"@"+config.postgres.host+":5432/";
    }
    var client = new pg.Client(conString);
    client.connect();
    return client;
}

function sql(opt) {
    
    if(opt === "login"){
        return login;
    }
    if(opt === "update_resume"){
        return update_resume;
    }
}
module.exports.sql=sql;
module.exports.dbconnect = db;