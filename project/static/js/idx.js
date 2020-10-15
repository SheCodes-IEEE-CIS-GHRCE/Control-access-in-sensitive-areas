var loadFile = function(event) {
    if (document.getElementById("file").value != "") {

        var reader = new FileReader();
        reader.onload = function() {
            var output = document.getElementById('output');

            output.src = reader.result;

        };
        reader.readAsDataURL(event.target.files[0]);
    }

};

function myFunction() {
    var selected_file = document.getElementById("myfile");
    //console.log(typeof(selected_file));
    //console.log(selected_file);
    var selected_file_path = selected_file.value;
    console.log(selected_file_path);

    console.log("image uploaded..");
}
Attachments area
