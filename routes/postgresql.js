//import database.js
var db = require("./db.js");
//TODO
//var bcrypt = require("bcrypt");
//import fs module
var fs = require("fs");

//var jtrans = require("./jtrans")
//to hash
//bcrypt.hash(myPlaintextPassword, saltRounds, function(err, hash) {
//Store hash in your password DB.
//});
var sync = require("synchronize");

//module to export functions in postgresql.jss in postgresql.js
module.exports = {
/*
loginhash : function(userid,pass){
    var client = db.dbconnect('history');
    var valid = false;
    var query = client.query("select * from login where userid=($1)",[userid]);
    query.on('row', function(val) {
        bcrypt.compare(pass,val['passwd'],function(err,result){
            if(err){
                return err;
            }
            else{
                valid=result;
            }
        });
        val['valid']=valid;
        return val;
    });
},
prin : function(){
  console.log("hello ");  
},*/
    login : function(userid,pass,usercall){
        var client = db.dbconnect('studdb');
        var valout={};
        var login_sql = db.sql('login');
        var query = client.query(login_sql,[userid,pass]);
        query.on('row', function(val) {
            val['valid']=true;
            valout = val;
        });
        query.on('end',function(val) {
            client.end();
            usercall(valout);
        });
    }
    ,
    update_resume : function(userid,data,usercall){
        var client = db.dbconnect('studdb');
        var update_sql = db.sql('update_resume');
        var query = client.query(update_sql,[data.resume,userid]);
        query.on('end',function(val) {
            var valout=true
            console.log(val);
            client.end();
            usercall(valout);
        });
    }
};