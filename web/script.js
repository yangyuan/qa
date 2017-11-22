$(document).ready(function() {
});


function launch() {
    $("#qa-out").empty();
    $("#qa-out").append($('<div class="progress row"><div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100" style="width: 100%; height: 30px;"></div></div>'));

    ('')
    $.ajax({
        type: 'POST',
        url: '/qa',
        data: { 'passage': $('#passage').val(), 'question': $('#question').val() },
        dataType: 'json'}
        ).done(function(data) {
            $("#qa-out").empty();
            var tmp = $('<div></div>').addClass("alert alert-success row").append(data["answer"]);
            $("#qa-out").append(tmp);
        }).fail(function(xhr, textStatus, errorThrown){
            $("#qa-out").empty();
            var tmp = $('<div></div>').addClass("alert alert-danger row").append(errorThrown);
            $("#qa-out").append(tmp);
        });
}