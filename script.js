$(document).ready(function() {
    // Character count update
    $('#inputText').on('input', function() {
        const inputText = $(this).val();
        $('#charCount').text(`Characters: ${inputText.length}`);
    });

    // Theme switcher
    $('input[name="theme"]').on('change', function() {
        if ($('#lightTheme').is(':checked')) {
            $('body').removeClass('bg-dark').addClass('bg-light');
            $('textarea').removeClass('bg-dark text-light').addClass('bg-light text-dark');
            $('.navbar').removeClass('bg-dark').addClass('bg-light navbar-light');
            $('.navbar-brand').addClass('light-mode');
            $('.footer').removeClass('bg-dark').addClass('bg-light text-dark');
            $('h3').addClass('light-mode');
        } else {
            $('body').removeClass('bg-light').addClass('bg-dark');
            $('textarea').removeClass('bg-light text-dark').addClass('bg-dark text-light');
            $('.navbar').removeClass('bg-light navbar-light').addClass('bg-dark navbar-dark');
            $('.navbar-brand').removeClass('light-mode');
            $('.footer').removeClass('bg-light text-dark').addClass('bg-dark text-light');
            $('h3').removeClass('light-mode');
        }
    });

    // Text summarization
    const summarizeText = () => {
        const inputText = $('#inputText').val();
        const method = $('#abstractive').is(':checked') ? 'abstractive' : 'extractive';
        
        $('#loadingSpinner').show();
        
        $.ajax({
            url: 'http://127.0.0.1:8000/summarize',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ text: inputText, method: method }),
            success: function(response) {
                $('#summaryText').val(response.summary);
                $('#loadingSpinner').hide();
            },
            error: function() {
                alert('Error summarizing text.');
                $('#loadingSpinner').hide();
            }
        });
    };

    // Expose summarizeText function globally
    window.summarizeText = summarizeText;
});
